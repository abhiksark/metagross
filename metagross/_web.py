# metagross/_web.py
"""Local, dependency-free browser dashboard for Metagross traces."""

from __future__ import annotations

import datetime
import hmac
import http.server
import ipaddress
import json
import secrets
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from metagross import _follow, _tui, _viewer


_DEFAULT_PORT = 8765
_DEFAULT_HOST = "127.0.0.1"
_TOP_LIMIT = 8
_RECENT_LIMIT = 50
_MEMORY_LIMIT = 120
_TIMELINE_LIMIT = 1_000
_MAX_CONTROL_BODY = 4 << 10
_MAX_EVENT_BODY = 1 << 20
_MAX_FINISH_BODY = _viewer._MAX_SUMMARY_BYTES + (64 << 10)
_MAX_RESPONSE_BODY = 4 << 10
_MAX_BATCH_EVENTS = 128
# Keep these paths aligned with assets/logo-mark.svg.
# Embed the artwork so copied packages and Docker images need no asset tree.
_LOGO_PATHS = b"""<path d="M303.46 50.83 L224.23 132.30 L287.77 196.58 L344.57 145.00 L415.58 215.26 L415.58 241.42 L372.98 284.03 L438.00 350.55 L512.00 276.55 L512.00 167.43 L396.15 50.83 Z"/>
<path d="M28.40 0.00 L28.40 42.60 L458.93 480.61 L458.93 412.59 L178.64 123.33 L214.52 85.96 L129.31 0.00 Z"/>
<path d="M84.46 241.42 L0.75 325.14 L0.75 390.17 L119.59 512.00 L202.56 512.00 L276.55 436.51 L214.52 372.98 L169.67 417.82 L162.20 417.82 L98.66 354.29 L98.66 346.07 L143.51 299.73 Z"/>
<path d="M29.90 80.72 L29.90 148.74 L387.18 512.00 L452.95 512.00 Z"/>"""
_LOGO_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
    b'<title>Metagross</title><g fill="#8ac926">' + _LOGO_PATHS + b'</g></svg>'
)
_FAVICON_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
    b'<title>Metagross</title><rect width="512" height="512" fill="#202020"/>'
    b'<g transform="translate(24 24) scale(0.90625)" fill="#8ac926">'
    + _LOGO_PATHS + b'</g></svg>'
)

_INDEX_HTML = b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Local CUDA driver trace analysis for Metagross">
  <title>Metagross Trace Analysis</title>
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/app.css">
</head>
<body>
  <a class="skip-link" href="#workspace">Skip to trace workspace</a>
  <header class="app-header">
    <div class="brand">
      <img class="brand-mark" src="/logo.svg" width="32" height="32" alt="">
      <span class="brand-name">METAGROSS</span>
      <span class="brand-separator"></span>
      <span class="brand-product">Trace Analysis</span>
    </div>
    <div class="header-trace">
      <span class="header-label">Report</span>
      <strong id="trace-title">Waiting for trace</strong>
    </div>
    <div class="header-actions">
      <span id="updated-at" class="updated-at">Connecting</span>
      <button id="refresh-button" class="tool-button" type="button">Refresh</button>
      <button id="pause-button" class="tool-button primary" type="button" aria-pressed="false">Pause</button>
    </div>
  </header>

  <nav class="view-tabs" aria-label="Analysis views">
    <button class="view-tab active" type="button" aria-current="page">Timeline</button>
    <a class="view-tab" href="#summary-analysis">Summary</a>
    <span class="view-context">CPU-side CUDA driver activity</span>
    <div id="status-badge" class="status-badge waiting">
      <span class="status-dot" aria-hidden="true"></span>
      <span id="status-text">WAITING</span>
    </div>
  </nav>

  <main id="workspace" class="workspace">
    <div id="warning" class="warning" role="status" hidden></div>
    <form id="token-form" class="token-form" hidden>
      <label for="token-input">Viewer token or private dashboard URL</label>
      <input id="token-input" type="text" autocomplete="off" spellcheck="false" required>
      <button class="tool-button primary" type="submit">Connect</button>
    </form>

    <section class="metric-strip" aria-label="Trace summary">
      <article class="metric-cell signal">
        <span>Events</span><strong id="metric-events">0</strong>
        <small id="metric-rate">0.0 /s</small>
      </article>
      <article class="metric-cell">
        <span>Attributed</span><strong id="metric-attributed">0.0%</strong>
        <small>project frames</small>
      </article>
      <article class="metric-cell">
        <span>CUDA errors</span><strong id="metric-errors">0</strong>
        <small>return codes</small>
      </article>
      <article class="metric-cell">
        <span>CPU API</span><strong id="metric-cpu">0ns</strong>
        <small>total duration</small>
      </article>
      <article class="metric-cell">
        <span>Sync</span><strong id="metric-sync">0ns</strong>
        <small>blocking time</small>
      </article>
      <article class="metric-cell">
        <span>Copied</span><strong id="metric-copied">0B</strong>
        <small>successful</small>
      </article>
      <article class="metric-cell">
        <span>GPU observed</span><strong id="metric-observed">0B</strong>
        <small>outstanding</small>
      </article>
      <article class="metric-cell">
        <span>Observed peak</span><strong id="metric-peak">0B</strong>
        <small>allocation high</small>
      </article>
    </section>

    <section class="analysis-panel timeline-panel" aria-labelledby="timeline-heading">
      <header class="panel-titlebar">
        <div>
          <span class="section-index">01</span>
          <h1 id="timeline-heading">CUDA API timeline</h1>
          <span id="timeline-range" class="title-meta">No timed events</span>
        </div>
        <div class="integrity" aria-label="Capture integrity">
          <span>Lost <strong id="metric-lost">0</strong></span>
          <span>Dropped <strong id="metric-dropped">0</strong></span>
          <span>Delivery <strong id="metric-delivery-dropped">0</strong></span>
          <span>Malformed <strong id="metric-malformed">0</strong></span>
        </div>
      </header>

      <div class="analysis-toolbar">
        <label class="search-control">
          <span class="sr-only">Search timeline</span>
          <input id="trace-search" type="search" autocomplete="off" placeholder="Filter API, function, kernel, span">
        </label>
        <div class="family-filters" aria-label="CUDA API family filters">
          <button class="family-filter launch active" data-family="launch" type="button" aria-pressed="true">Launch</button>
          <button class="family-filter copy active" data-family="copy" type="button" aria-pressed="true">Copy</button>
          <button class="family-filter memory active" data-family="memory" type="button" aria-pressed="true">Memory</button>
          <button class="family-filter sync active" data-family="sync" type="button" aria-pressed="true">Sync</button>
          <button class="family-filter other active" data-family="other" type="button" aria-pressed="true">Other</button>
        </div>
        <div class="zoom-tools" aria-label="Timeline zoom controls">
          <button id="zoom-out" class="icon-button" type="button" aria-label="Zoom out">-</button>
          <button id="zoom-reset" class="zoom-readout" type="button">100%</button>
          <button id="zoom-in" class="icon-button" type="button" aria-label="Zoom in">+</button>
        </div>
      </div>

      <div class="timeline-workspace">
        <div class="lane-column" aria-hidden="true">
          <div class="lane-header">Project function</div>
          <div id="lane-labels" class="lane-labels"></div>
        </div>
        <div id="timeline-scroll" class="timeline-scroll" tabindex="0" aria-label="Scrollable CUDA API timeline">
          <div id="timeline-canvas" class="timeline-canvas">
            <div id="timeline-ruler" class="timeline-ruler"></div>
            <div id="timeline-lanes" class="timeline-lanes"></div>
          </div>
          <p id="timeline-empty" class="timeline-empty">Waiting for timed CUDA events.</p>
        </div>
      </div>
      <footer class="timeline-footer">
        <div class="legend" aria-label="Timeline legend">
          <span class="launch">Launch</span><span class="copy">Copy</span>
          <span class="memory">Memory</span><span class="sync">Sync</span>
          <span class="error">CUDA error</span>
        </div>
        <span>Drag to pan - select an event for details</span>
      </footer>
    </section>
    <div id="timeline-tooltip" class="timeline-tooltip" role="tooltip" hidden></div>

    <section class="detail-grid">
      <aside class="analysis-panel inspector" aria-labelledby="inspector-heading">
        <header class="panel-titlebar compact">
          <div><span class="section-index">02</span><h2 id="inspector-heading">Selection details</h2></div>
        </header>
        <div id="selection-empty" class="inspector-empty">
          <strong>No event selected</strong>
          <span>Select a timeline range or event-table row.</span>
        </div>
        <div id="selection-content" class="selection-content" hidden>
          <div class="selection-head">
            <span id="detail-family" class="family-badge">API</span>
            <strong id="detail-api">-</strong>
            <span id="detail-result" class="result-badge">OK</span>
          </div>
          <section class="detail-section">
            <h3>Timing</h3>
            <dl>
              <div><dt>CPU duration</dt><dd id="detail-duration">-</dd></div>
              <div><dt>Timestamp</dt><dd id="detail-timestamp">-</dd></div>
              <div><dt>Process / thread</dt><dd id="detail-thread">-</dd></div>
            </dl>
          </section>
          <section class="detail-section">
            <h3>Source correlation</h3>
            <dl>
              <div><dt>Function</dt><dd id="detail-function">-</dd></div>
              <div><dt>Location</dt><dd id="detail-location">-</dd></div>
              <div><dt>Kernel</dt><dd id="detail-kernel">-</dd></div>
              <div><dt>Span</dt><dd id="detail-span">-</dd></div>
            </dl>
          </section>
          <section class="detail-section">
            <h3>Captured arguments</h3>
            <pre id="detail-json">{}</pre>
          </section>
        </div>
      </aside>

      <article class="analysis-panel event-view" aria-labelledby="events-heading">
        <header class="panel-titlebar compact">
          <div>
            <span class="section-index">03</span><h2 id="events-heading">Events view</h2>
            <span id="event-count" class="title-meta">0 visible</span>
          </div>
        </header>
        <p class="scroll-hint">Scroll horizontally for API, kernel, duration, and result.</p>
        <div class="table-scroll">
          <table>
            <thead><tr><th>Time</th><th>Function</th><th>CUDA API</th><th>Kernel</th><th>CPU</th><th>Result</th></tr></thead>
            <tbody id="recent-events"><tr><td colspan="6" class="empty">Waiting for events</td></tr></tbody>
          </table>
        </div>
      </article>
    </section>

    <section id="summary-analysis" class="summary-analysis" aria-labelledby="summary-heading">
      <header class="summary-heading">
        <div><span class="section-index">04</span><h2 id="summary-heading">Summary analysis</h2></div>
        <span>Compute-style aggregate sections</span>
      </header>
      <div class="summary-grid">
        <article class="analysis-panel summary-card">
          <header><h3>Top CUDA APIs</h3><span>Calls / CPU total</span></header>
          <div id="top-apis" class="rank-list"><p class="empty">Waiting for CUDA events</p></div>
        </article>
        <article class="analysis-panel summary-card">
          <header><h3>Project functions</h3><span>Resolved frames</span></header>
          <div id="top-functions" class="rank-list"><p class="empty">No attributed functions yet</p></div>
        </article>
        <article class="analysis-panel summary-card">
          <header><h3>Top kernels</h3><span>Resolved launches</span></header>
          <div id="top-kernels" class="rank-list"><p class="empty">No resolved kernels yet</p></div>
        </article>
        <article class="analysis-panel summary-card memory-panel">
          <header><h3>Observed allocation</h3><span id="memory-range">No samples</span></header>
          <div class="chart-wrap">
            <svg id="memory-chart" viewBox="0 0 500 150" role="img" aria-label="Observed GPU allocation over time" preserveAspectRatio="none">
              <path class="chart-grid" d="M0 37.5H500M0 75H500M0 112.5H500"></path>
              <path id="memory-area" class="chart-area" d=""></path>
              <path id="memory-line" class="chart-line" d=""></path>
            </svg>
            <p id="memory-empty" class="empty chart-empty">Allocation samples appear when memory APIs are traced.</p>
          </div>
        </article>
      </div>
    </section>
  </main>

  <footer class="footer">
    <span>Metagross local trace analysis</span>
    <span id="footer-refresh">Refresh 0.20s</span>
  </footer>
  <script src="/app.js" defer></script>
</body>
</html>
"""

_APP_CSS = b""":root {
  color-scheme: dark;
  --app-bg: #171717;
  --chrome: #202020;
  --panel: #252525;
  --panel-high: #2b2b2b;
  --panel-low: #1c1c1c;
  --line: #424242;
  --line-soft: #343434;
  --text: #f0f0f0;
  --muted: #b3b3b3;
  --dim: #969696;
  --signal: #8ac926;
  --signal-soft: #9bc95c;
  --launch: #8ac926;
  --copy: #c96e62;
  --memory: #6293b8;
  --sync: #c79d55;
  --other: #8a8a8a;
  --error: #e05a4f;
  --sans: "Segoe UI", "Noto Sans", Arial, sans-serif;
  --mono: "SFMono-Regular", "Cascadia Mono", "Roboto Mono", Consolas, monospace;
  --lane-height: 40px;
  --ruler-height: 34px;
}

* { box-sizing: border-box; }
[hidden] { display: none !important; }
html { background: var(--app-bg); scroll-behavior: smooth; }
body {
  min-height: 100dvh;
  margin: 0;
  color: var(--text);
  background:
    linear-gradient(rgba(255, 255, 255, 0.012) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.009) 1px, transparent 1px),
    var(--app-bg);
  background-size: 4px 4px;
  font-family: var(--sans);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}

button, input { font: inherit; }
button { color: inherit; }

.skip-link {
  position: fixed;
  z-index: 20;
  top: 8px;
  left: 8px;
  padding: 8px 10px;
  color: #101010;
  background: var(--signal);
  transform: translateY(-150%);
}
.skip-link:focus { transform: translateY(0); }
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}

.app-header {
  position: sticky;
  z-index: 10;
  top: 0;
  display: grid;
  grid-template-columns: minmax(240px, 1fr) minmax(220px, 1fr) minmax(240px, 1fr);
  align-items: center;
  min-height: 48px;
  padding: 0 16px;
  border-bottom: 1px solid #090909;
  background: #191919;
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04);
}
.brand, .header-actions, .header-trace { display: flex; align-items: center; }
.brand { gap: 9px; }
.brand-name {
  font-family: "DejaVu Sans", var(--sans);
  font-size: 12px;
  font-weight: 400;
  letter-spacing: 0.18em;
}
.brand-product { color: var(--muted); font-size: 11px; }
.brand-separator { width: 1px; height: 18px; margin: 0 3px; background: var(--line); }
.brand-mark {
  display: block;
  width: 32px;
  height: 32px;
  flex: none;
}
.header-trace { justify-content: center; gap: 9px; min-width: 0; }
.header-trace strong {
  max-width: 360px;
  overflow: hidden;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.header-label {
  color: var(--dim);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.header-actions { justify-content: flex-end; gap: 6px; }
.updated-at { margin-right: 8px; color: var(--dim); font-family: var(--mono); font-size: 11px; }
.tool-button, .icon-button, .zoom-readout {
  min-height: 27px;
  border: 1px solid #5c5c5c;
  border-radius: 2px;
  color: #d0d0d0;
  background: #2a2a2a;
  cursor: pointer;
  transition: border-color 120ms ease, background-color 120ms ease, color 120ms ease;
}
.tool-button { padding: 4px 11px; }
.tool-button.primary { border-color: #5e8f0b; color: #eaf6d8; background: #314300; }
.tool-button:hover, .icon-button:hover, .zoom-readout:hover { border-color: #6a6a6a; background: #353535; }
.tool-button.primary:hover { border-color: var(--signal); background: #3d5300; }
.tool-button:active, .icon-button:active, .zoom-readout:active { transform: translateY(1px); }
button:focus-visible, input:focus-visible, a:focus-visible, .timeline-scroll:focus-visible {
  outline: 2px solid var(--signal);
  outline-offset: 2px;
}
.skip-link:focus-visible { outline-color: #fff; }

.view-tabs {
  display: flex;
  align-items: center;
  height: 38px;
  padding: 0 16px;
  border-bottom: 1px solid #101010;
  background: var(--chrome);
}
.view-tab {
  align-self: stretch;
  display: grid;
  place-items: center;
  min-width: 88px;
  padding: 0 14px;
  border: 0;
  border-right: 1px solid #303030;
  color: var(--muted);
  background: transparent;
  font-size: 12px;
  text-decoration: none;
}
.view-tab:first-child { border-left: 1px solid #303030; }
.view-tab.active { color: var(--text); background: var(--panel); box-shadow: inset 0 -2px var(--signal); }
.view-tab:hover { color: var(--text); background: #292929; }
.view-context { margin-left: 14px; color: var(--dim); font-size: 11px; }
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  margin-left: auto;
  color: var(--muted);
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.status-badge.live, .status-badge.complete { color: var(--signal-soft); }
.status-badge.incomplete, .status-badge.mismatch, .status-badge.malformed { color: var(--sync); }
.status-badge.error { color: var(--error); }

.workspace { width: 100%; padding: 12px; }
.warning {
  margin-bottom: 10px;
  padding: 9px 11px;
  border-left: 3px solid var(--sync);
  color: #ead2a5;
  background: #392f1f;
  font-family: var(--mono);
  font-size: 11px;
}
.token-form {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  color: var(--muted);
  font-size: 12px;
}
.token-form input {
  flex: 1 1 280px;
  height: 27px;
  padding: 4px 9px;
  border: 1px solid #5c5c5c;
  border-radius: 2px;
  color: var(--text);
  background: #1c1c1c;
  font-family: var(--mono);
  font-size: 12px;
}

.metric-strip {
  display: grid;
  grid-template-columns: repeat(8, minmax(110px, 1fr));
  margin-bottom: 10px;
  border: 1px solid var(--line);
  background: var(--line);
  gap: 1px;
  overflow-x: auto;
}
.metric-cell { min-width: 110px; padding: 9px 11px 8px; background: var(--panel); }
.metric-cell.signal { box-shadow: inset 3px 0 var(--signal); background: #293019; }
.metric-cell span { display: block; color: var(--muted); font-size: 10px; letter-spacing: 0.07em; text-transform: uppercase; }
.metric-cell strong { display: block; margin-top: 5px; font-family: var(--mono); font-size: 18px; font-weight: 500; line-height: 1; }
.metric-cell.signal strong { color: var(--signal-soft); }
.metric-cell small { display: block; margin-top: 5px; color: var(--dim); font-size: 10px; }

.analysis-panel { min-width: 0; border: 1px solid var(--line); background: var(--panel); }
.panel-titlebar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 43px;
  padding: 7px 10px;
  border-bottom: 1px solid var(--line);
  background: var(--panel-high);
}
.panel-titlebar > div, .summary-heading > div { display: flex; align-items: baseline; gap: 9px; min-width: 0; }
.panel-titlebar h1, .panel-titlebar h2, .summary-heading h2 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0;
}
.panel-titlebar h1 { font-size: 15px; }
.panel-titlebar.compact { min-height: 40px; }
.section-index { color: var(--signal); font-family: var(--mono); font-size: 10px; font-weight: 700; }
.title-meta { overflow: hidden; color: var(--dim); font-family: var(--mono); font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
.integrity { display: flex; gap: 13px; color: var(--dim); font-family: var(--mono); font-size: 10px; }
.integrity strong { color: #d2d2d2; font-weight: 500; }

.analysis-toolbar {
  display: grid;
  grid-template-columns: minmax(210px, 1fr) auto auto;
  align-items: center;
  gap: 10px;
  min-height: 40px;
  padding: 6px 8px;
  border-bottom: 1px solid var(--line);
  background: var(--chrome);
}
.search-control input {
  width: min(360px, 100%);
  height: 27px;
  padding: 4px 9px 4px 27px;
  border: 1px solid #5c5c5c;
  border-radius: 2px;
  color: var(--text);
  background:
    linear-gradient(45deg, transparent 47%, #858585 48% 52%, transparent 53%) 10px 16px / 7px 7px no-repeat,
    radial-gradient(circle, transparent 45%, #858585 48% 57%, transparent 60%) 7px 6px / 11px 11px no-repeat,
    #1c1c1c;
  font-size: 12px;
}
.search-control input::placeholder { color: var(--dim); }
.family-filters, .zoom-tools { display: flex; align-items: center; gap: 3px; }
.family-filter {
  min-height: 29px;
  padding: 4px 8px;
  border: 1px solid #555;
  border-radius: 2px;
  color: var(--dim);
  background: #242424;
  font-size: 10px;
  cursor: pointer;
}
.family-filter::before { display: inline-block; width: 6px; height: 6px; margin-right: 5px; background: currentColor; content: ""; }
.family-filter.active { color: #e0e0e0; background: #333; }
.family-filter.launch.active::before { background: var(--launch); }
.family-filter.copy.active::before { background: var(--copy); }
.family-filter.memory.active::before { background: var(--memory); }
.family-filter.sync.active::before { background: var(--sync); }
.family-filter.other.active::before { background: var(--other); }
.family-filter:not(.active) { opacity: 0.72; }
.icon-button { width: 29px; padding: 0; font-family: var(--mono); font-size: 16px; }
.zoom-readout { width: 56px; padding: 0; font-family: var(--mono); font-size: 10px; }

.timeline-workspace { display: grid; grid-template-columns: 210px minmax(0, 1fr); min-height: 250px; max-height: 430px; background: var(--panel-low); }
.lane-column { z-index: 3; overflow: hidden; border-right: 1px solid var(--line); background: #222; box-shadow: 4px 0 8px rgba(0, 0, 0, 0.18); }
.lane-header {
  display: flex;
  align-items: center;
  height: var(--ruler-height);
  padding: 0 10px;
  border-bottom: 1px solid var(--line);
  color: var(--dim);
  background: #1d1d1d;
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.lane-labels { will-change: transform; }
.lane-label {
  display: flex;
  align-items: center;
  height: var(--lane-height);
  padding: 0 10px;
  border-bottom: 1px solid var(--line-soft);
  overflow: hidden;
  color: #c6c6c6;
  font-family: var(--mono);
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.lane-label::before { margin-right: 7px; color: var(--dim); content: "fn"; font-size: 9px; }
.timeline-scroll {
  position: relative;
  min-width: 0;
  overflow: auto;
  scrollbar-color: #5b5b5b #202020;
  scrollbar-width: thin;
}
.timeline-canvas { position: relative; min-width: 100%; min-height: 100%; }
.timeline-ruler {
  position: sticky;
  z-index: 2;
  top: 0;
  height: var(--ruler-height);
  border-bottom: 1px solid var(--line);
  background: #1d1d1d;
}
.ruler-tick { position: absolute; inset: 0 auto 0 0; border-left: 1px solid #555; color: var(--dim); font-family: var(--mono); font-size: 10px; }
.ruler-tick span { position: absolute; top: 8px; left: 5px; white-space: nowrap; }
.timeline-lane { position: relative; height: var(--lane-height); border-bottom: 1px solid var(--line-soft); background-image: linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px); background-size: 10% 100%; }
.timeline-lane:nth-child(even) { background-color: rgba(255, 255, 255, 0.012); }
.event-block {
  position: absolute;
  top: 7px;
  height: 26px;
  min-width: 4px;
  overflow: hidden;
  padding: 3px 5px;
  border: 1px solid rgba(255, 255, 255, 0.24);
  border-radius: 1px;
  color: #f1f1f1;
  font-family: var(--mono);
  font-size: 10px;
  line-height: 18px;
  text-align: left;
  text-overflow: ellipsis;
  white-space: nowrap;
  cursor: pointer;
}
.event-block.launch { background: #4f7908; }
.event-block.copy { background: #83443c; }
.event-block.memory { background: #3e627d; }
.event-block.sync { color: #1c160b; background: #af873f; }
.event-block.other { background: #5a5a5a; }
.event-block.event-error { border-color: #ff9d94; background: #9f3028; }
.event-block:hover { z-index: 1; filter: brightness(1.18); }
.event-block.selected { z-index: 2; outline: 2px solid #fff; outline-offset: 1px; }
.timeline-tooltip {
  position: fixed;
  z-index: 20;
  max-width: min(440px, calc(100vw - 16px));
  padding: 7px 9px;
  border: 1px solid #686868;
  color: var(--text);
  background: #101010;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.45);
  font-family: var(--mono);
  font-size: 11px;
  line-height: 1.45;
  pointer-events: none;
}
.timeline-empty { position: absolute; inset: 34px 0 0; display: grid; place-items: center; margin: 0; color: var(--dim); font-family: var(--mono); font-size: 11px; pointer-events: none; }
.timeline-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 31px;
  padding: 5px 9px;
  border-top: 1px solid var(--line);
  color: var(--dim);
  background: var(--chrome);
  font-size: 10px;
}
.legend { display: flex; flex-wrap: wrap; gap: 12px; }
.legend span::before { display: inline-block; width: 8px; height: 8px; margin-right: 5px; content: ""; }
.legend .launch::before { background: var(--launch); }
.legend .copy::before { background: var(--copy); }
.legend .memory::before { background: var(--memory); }
.legend .sync::before { background: var(--sync); }
.legend .error::before { background: var(--error); }

.detail-grid { display: grid; grid-template-columns: minmax(300px, 0.34fr) minmax(0, 0.66fr); gap: 10px; margin-top: 10px; }
.inspector, .event-view { min-height: 355px; }
.inspector-empty { display: grid; place-content: center; min-height: 305px; padding: 24px; color: var(--dim); text-align: center; }
.inspector-empty strong { margin-bottom: 5px; color: #d0d0d0; font-weight: 500; }
.selection-content { padding: 9px; }
.selection-head { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center; gap: 8px; padding: 3px 2px 10px; }
.selection-head > strong { overflow: hidden; font-family: var(--mono); font-size: 13px; font-weight: 600; text-overflow: ellipsis; white-space: nowrap; }
.family-badge, .result-badge { padding: 2px 5px; border: 1px solid #666; color: #d8d8d8; background: #313131; font-family: var(--mono); font-size: 9px; text-transform: uppercase; }
.result-badge.ok { border-color: #6f971a; color: var(--signal-soft); }
.result-badge.error { border-color: #a84940; color: #f0958d; }
.detail-section { margin-top: 8px; border: 1px solid var(--line); background: #212121; }
.detail-section h3 { margin: 0; padding: 7px 8px; border-bottom: 1px solid var(--line); color: #dedede; background: #2b2b2b; font-size: 11px; font-weight: 600; }
.detail-section dl { margin: 0; }
.detail-section dl > div { display: grid; grid-template-columns: 112px minmax(0, 1fr); gap: 10px; padding: 7px 8px; border-top: 1px solid #353535; }
.detail-section dl > div:first-child { border-top: 0; }
.detail-section dt { color: var(--dim); font-size: 11px; }
.detail-section dd { overflow: hidden; margin: 0; color: #dedede; font-family: var(--mono); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
.detail-section pre { max-height: 128px; margin: 0; overflow: auto; padding: 8px; color: #c8d6b4; font-family: var(--mono); font-size: 11px; line-height: 1.5; white-space: pre-wrap; }

.table-scroll { max-height: 315px; overflow: auto; scrollbar-color: #5b5b5b #202020; scrollbar-width: thin; }
table { width: 100%; min-width: 760px; border-collapse: collapse; table-layout: fixed; }
thead { position: sticky; z-index: 2; top: 0; background: #202020; }
th { height: 31px; padding: 0 8px; border-bottom: 1px solid var(--line); color: var(--dim); font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-align: left; text-transform: uppercase; }
td { height: 32px; overflow: hidden; padding: 0 8px; border-bottom: 1px solid var(--line-soft); color: #d0d0d0; font-family: var(--mono); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
tbody tr { cursor: pointer; }
tbody tr:hover td, tbody tr:focus-visible td { background: #303030; }
tbody tr:focus-visible { outline: 2px solid var(--signal); outline-offset: -2px; }
tbody tr.selected td { background: #35431f; box-shadow: inset 0 1px #67872c, inset 0 -1px #67872c; }
tbody tr.event-error td { color: #e8a39d; }
th:nth-child(1), td:nth-child(1) { position: sticky; left: 0; z-index: 1; width: 14%; background: var(--panel); }
thead th:nth-child(1) { z-index: 3; background: #202020; }
tbody tr:hover td:nth-child(1), tbody tr:focus-visible td:nth-child(1) { background: #303030; }
tbody tr.selected td:nth-child(1) { background: #35431f; }
th:nth-child(2), td:nth-child(2) { width: 20%; }
th:nth-child(3), td:nth-child(3) { width: 22%; }
th:nth-child(4), td:nth-child(4) { width: 22%; }
th:nth-child(5), td:nth-child(5) { width: 12%; text-align: right; }
th:nth-child(6), td:nth-child(6) { width: 10%; text-align: right; }
.result-ok { color: var(--signal-soft); }
.result-error { color: #f0958d; }
.empty { color: var(--dim); font-family: var(--mono); font-size: 11px; }
.scroll-hint { display: none; margin: 0; padding: 7px 9px; border-bottom: 1px solid var(--line); color: var(--dim); background: var(--panel-low); font-size: 11px; }

.summary-analysis { margin-top: 18px; scroll-margin-top: 96px; }
.summary-heading { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; color: var(--dim); font-size: 10px; }
.summary-heading h2 { color: var(--text); }
.summary-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.summary-card { padding-bottom: 9px; }
.summary-card > header { display: flex; align-items: center; justify-content: space-between; min-height: 39px; padding: 0 9px; border-bottom: 1px solid var(--line); background: var(--panel-high); }
.summary-card h3 { margin: 0; font-size: 12px; font-weight: 600; }
.summary-card header span { color: var(--dim); font-family: var(--mono); font-size: 10px; }
.rank-list { display: grid; gap: 1px; min-height: 150px; padding: 7px; align-content: start; }
.rank-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; align-items: center; min-height: 27px; }
.rank-main { position: relative; min-width: 0; padding: 4px 6px; overflow: hidden; }
.rank-bar { position: absolute; inset: 0 auto 0 0; min-width: 2px; background: rgba(118, 185, 0, 0.12); border-left: 2px solid rgba(118, 185, 0, 0.48); }
.rank-label, .rank-sub { position: relative; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rank-label { color: #dedede; font-family: var(--mono); font-size: 11px; }
.rank-sub { margin-top: 3px; color: var(--dim); font-size: 10px; }
.rank-value { color: var(--dim); font-family: var(--mono); font-size: 10px; text-align: right; white-space: nowrap; }
.rank-value strong { display: block; color: #e0e0e0; font-size: 11px; font-weight: 500; }
.rank-list > .empty { margin: auto; }
.chart-wrap { position: relative; min-height: 150px; padding: 7px; }
#memory-chart { display: block; width: 100%; height: 150px; }
.chart-grid { fill: none; stroke: #383838; stroke-width: 1; vector-effect: non-scaling-stroke; }
.chart-area { fill: rgba(118, 185, 0, 0.12); }
.chart-line { fill: none; stroke: var(--signal); stroke-width: 1.5; vector-effect: non-scaling-stroke; }
.chart-empty { position: absolute; inset: 0; display: grid; place-items: center; margin: 0; text-align: center; }

.footer { display: flex; justify-content: space-between; padding: 12px; color: var(--dim); font-family: var(--mono); font-size: 11px; }

@media (max-width: 1050px) {
  .app-header { grid-template-columns: minmax(0, 1fr) auto; padding-block: 6px; }
  .header-trace { grid-column: 1 / -1; grid-row: 2; justify-content: flex-start; min-height: 24px; }
  .analysis-toolbar { grid-template-columns: 1fr auto; }
  .family-filters { grid-column: 1 / -1; grid-row: 2; }
  .metric-strip { grid-template-columns: repeat(4, minmax(120px, 1fr)); }
}

@media (max-width: 760px) {
  .app-header { padding: 0 9px; }
  .brand-product, .updated-at, .view-context { display: none; }
  .view-tabs { padding: 0 9px; }
  .workspace { padding: 8px; }
  .analysis-toolbar { grid-template-columns: minmax(0, 1fr) auto; }
  .family-filters { overflow-x: auto; }
  .timeline-workspace { grid-template-columns: 128px minmax(0, 1fr); }
  .lane-label { padding: 0 6px; font-size: 10px; }
  .lane-label::before { display: none; }
  .detail-grid, .summary-grid { grid-template-columns: 1fr; }
  .metric-strip { grid-template-columns: repeat(4, 128px); scroll-snap-type: x proximity; }
  .metric-cell { scroll-snap-align: start; }
  .timeline-footer { align-items: flex-start; flex-direction: column; gap: 5px; }
  .scroll-hint { display: block; }
}

@media (max-width: 460px) {
  .brand-separator { display: none; }
  .tool-button { min-width: 58px; padding-inline: 7px; }
  .family-filter { padding-inline: 7px; }
  .integrity { width: 100%; justify-content: flex-start; gap: 12px; }
  .timeline-panel .panel-titlebar { align-items: flex-start; flex-wrap: wrap; gap: 6px; }
  .timeline-panel .panel-titlebar > div:first-child { flex-wrap: wrap; }
  .timeline-panel .title-meta { flex-basis: 100%; line-height: 1.4; white-space: normal; }
}
"""

_APP_JS = rb"""(() => {
  "use strict";

  const tokenKey = "metagross-viewer-token";
  let viewerToken = "";
  function readViewerToken() {
    viewerToken = "";
    try {
      const tokens = new URLSearchParams(window.location.hash.slice(1)).getAll("viewer_token");
      if (tokens.length) {
        window.history.replaceState(null, "", window.location.pathname + window.location.search);
        window.sessionStorage.removeItem(tokenKey);
        if (tokens.length === 1 && tokens[0]) {
          window.sessionStorage.setItem(tokenKey, tokens[0]);
          viewerToken = tokens[0];
        }
      } else {
        viewerToken = window.sessionStorage.getItem(tokenKey) || "";
      }
    } catch {
      viewerToken = "";
    }
  }
  readViewerToken();
  window.addEventListener("hashchange", readViewerToken);

  // A pasted token takes the same session-storage path as the URL fragment,
  // but never enters the address bar or history.
  function usePastedToken(text) {
    const value = text.trim();
    const hash = value.indexOf("#");
    const tokens = hash < 0
      ? [value]
      : new URLSearchParams(value.slice(hash + 1)).getAll("viewer_token");
    if (tokens.length !== 1 || !/^[A-Za-z0-9_-]+$/.test(tokens[0])) return false;
    try {
      window.sessionStorage.setItem(tokenKey, tokens[0]);
    } catch {
      return false;
    }
    viewerToken = tokens[0];
    return true;
  }

  function tokenError(message) {
    return Object.assign(new Error(message), {needsToken: true});
  }

  const byId = (id) => document.getElementById(id);
  const pauseButton = byId("pause-button");
  const refreshButton = byId("refresh-button");
  const searchInput = byId("trace-search");
  const timelineScroll = byId("timeline-scroll");
  const timelineCanvas = byId("timeline-canvas");
  const laneLabels = byId("lane-labels");
  const timelineTooltip = byId("timeline-tooltip");
  const activeFamilies = new Set(["launch", "copy", "memory", "sync", "other"]);
  let paused = false;
  let timer = null;
  let refreshMs = 200;
  let latest = null;
  let zoom = 1;
  let selectedKey = null;
  let drag = null;
  let renderedTimelineSignature = null;
  let tooltipAnchor = null;

  const formatCount = (value) => Number(value || 0).toLocaleString();
  const setText = (id, value) => { byId(id).textContent = String(value); };
  const eventKey = (event) => `${latest?.generation || 0}:${event.id}`;

  function statusClass(status) {
    const value = status.toLowerCase();
    if (value.includes("error")) return "error";
    if (value.includes("mismatch")) return "mismatch";
    if (value.includes("incomplete")) return "incomplete";
    if (value.includes("malformed")) return "malformed";
    if (value.includes("complete")) return "complete";
    if (value.includes("live")) return "live";
    return "waiting";
  }

  function formatNanoseconds(value) {
    const ns = Number(value || 0);
    if (ns < 1000) return `${Math.round(ns)}ns`;
    if (ns < 1000000) return `${(ns / 1000).toFixed(ns < 10000 ? 1 : 0)}us`;
    if (ns < 1000000000) return `${(ns / 1000000).toFixed(ns < 10000000 ? 1 : 0)}ms`;
    return `${(ns / 1000000000).toFixed(ns < 10000000000 ? 2 : 1)}s`;
  }

  function matches(event) {
    if (!activeFamilies.has(event.family)) return false;
    const query = searchInput.value.trim().toLowerCase();
    if (!query) return true;
    return [event.api, event.function, event.kernel, event.file, event.span]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase().includes(query));
  }

  function visibleEvents() {
    return latest ? latest.timeline.events.filter(matches) : [];
  }

  function renderRankList(id, rows, emptyText, kind) {
    const container = byId(id);
    container.replaceChildren();
    if (!rows.length) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = emptyText;
      container.append(empty);
      return;
    }
    const maximum = Math.max(...rows.map((row) => Number(row.count)), 1);
    rows.forEach((row) => {
      const wrapper = document.createElement("div");
      wrapper.className = "rank-row";
      const main = document.createElement("div");
      main.className = "rank-main";
      const bar = document.createElement("div");
      bar.className = "rank-bar";
      bar.style.width = `${Math.max(2, Number(row.count) * 100 / maximum)}%`;
      const label = document.createElement("div");
      label.className = "rank-label";
      label.textContent = row.name;
      label.title = row.name;
      main.append(bar, label);
      if (kind === "function") {
        const sub = document.createElement("div");
        sub.className = "rank-sub";
        sub.textContent = row.file ? `${row.file}:${row.line || 0}` : "Source unavailable";
        sub.title = sub.textContent;
        main.append(sub);
      }
      const value = document.createElement("div");
      value.className = "rank-value";
      const count = document.createElement("strong");
      count.textContent = formatCount(row.count);
      const duration = document.createElement("span");
      duration.textContent = row.total;
      value.append(count, duration);
      wrapper.append(main, value);
      container.append(wrapper);
    });
  }

  function renderMemory(samples, formattedPeak) {
    const line = byId("memory-line");
    const area = byId("memory-area");
    const empty = byId("memory-empty");
    if (!samples.length) {
      line.setAttribute("d", "");
      area.setAttribute("d", "");
      empty.hidden = false;
      setText("memory-range", "No samples");
      return;
    }
    empty.hidden = true;
    const width = 500;
    const height = 150;
    const maximum = Math.max(...samples.map((sample) => Number(sample.bytes)), 1);
    const points = samples.map((sample, index) => {
      const x = samples.length === 1 ? width : index * width / (samples.length - 1);
      const y = height - (Number(sample.bytes) / maximum) * (height - 10);
      return [x, y];
    });
    const path = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(2)} ${y.toFixed(2)}`).join(" ");
    line.setAttribute("d", path);
    area.setAttribute("d", `${path} L${points.at(-1)[0].toFixed(2)} ${height} L${points[0][0].toFixed(2)} ${height} Z`);
    setText("memory-range", `${samples.length} samples / peak ${formattedPeak}`);
  }

  function eventDescription(event) {
    const parts = [
      event.api,
      `${event.duration} CPU`,
      event.function || "<unknown>",
    ];
    if (event.kernel) parts.push(event.kernel);
    parts.push(event.return_code ? `ERR ${event.return_code}` : "OK");
    return parts.join(" - ");
  }

  function hideEventTooltip(anchor) {
    if (anchor && document.activeElement === anchor) return;
    timelineTooltip.hidden = true;
    tooltipAnchor = null;
  }

  function positionEventTooltip() {
    if (!tooltipAnchor || timelineTooltip.hidden) return;
    const anchorBox = tooltipAnchor.getBoundingClientRect();
    const scrollerBox = timelineScroll.getBoundingClientRect();
    if (
      anchorBox.right < scrollerBox.left
      || anchorBox.left > scrollerBox.right
      || anchorBox.bottom < scrollerBox.top
      || anchorBox.top > scrollerBox.bottom
    ) {
      hideEventTooltip();
      return;
    }
    const tooltipBox = timelineTooltip.getBoundingClientRect();
    const left = Math.min(
      window.innerWidth - tooltipBox.width - 8,
      Math.max(8, anchorBox.left),
    );
    const below = anchorBox.bottom + 8;
    const top = below + tooltipBox.height <= window.innerHeight
      ? below
      : Math.max(8, anchorBox.top - tooltipBox.height - 8);
    timelineTooltip.style.left = `${left}px`;
    timelineTooltip.style.top = `${top}px`;
  }

  function showEventTooltip(anchor, event) {
    tooltipAnchor = anchor;
    timelineTooltip.textContent = eventDescription(event);
    timelineTooltip.hidden = false;
    requestAnimationFrame(positionEventTooltip);
  }

  function updateSelectedStyles() {
    document.querySelectorAll("[data-event-key]").forEach((node) => {
      const selected = node.dataset.eventKey === selectedKey;
      node.classList.toggle("selected", selected);
      if (node.matches(".event-block")) {
        node.setAttribute("aria-pressed", String(selected));
      } else if (node.matches("tr")) {
        node.setAttribute("aria-selected", String(selected));
      }
    });
  }

  function clearSelection() {
    selectedKey = null;
    byId("selection-empty").hidden = false;
    byId("selection-content").hidden = true;
    updateSelectedStyles();
  }

  function selectEvent(event) {
    selectedKey = eventKey(event);
    byId("selection-empty").hidden = true;
    byId("selection-content").hidden = false;
    setText("detail-family", event.family);
    setText("detail-api", event.api);
    setText("detail-duration", event.duration);
    setText("detail-timestamp", event.timestamp);
    setText("detail-thread", `${event.pid} / ${event.tid}`);
    setText("detail-function", event.function || "<unknown>");
    setText("detail-location", event.file ? `${event.file}:${event.line || 0}` : "<unknown>");
    setText("detail-kernel", event.kernel || "<not applicable>");
    setText("detail-span", event.span || "<none>");
    setText("detail-json", JSON.stringify(event.details, null, 2));
    const result = byId("detail-result");
    result.textContent = event.return_code ? `ERR ${event.return_code}` : "OK";
    result.className = `result-badge ${event.return_code ? "error" : "ok"}`;
    byId("detail-family").className = `family-badge ${event.family}`;
    updateSelectedStyles();
  }

  function renderRuler(spanNs) {
    const ruler = byId("timeline-ruler");
    ruler.replaceChildren();
    for (let index = 0; index <= 10; index += 1) {
      const tick = document.createElement("div");
      tick.className = "ruler-tick";
      tick.style.left = `${index * 10}%`;
      const label = document.createElement("span");
      label.textContent = formatNanoseconds(spanNs * index / 10);
      tick.append(label);
      ruler.append(tick);
    }
  }

  function renderTimeline(focusSelection = false) {
    if (!latest) return;
    const events = visibleEvents();
    hideEventTooltip();
    const spanNs = Math.max(1, Number(latest.timeline.span_ns));
    const oldWidth = Math.max(1, timelineScroll.scrollWidth);
    const center = (timelineScroll.scrollLeft + timelineScroll.clientWidth / 2) / oldWidth;
    const canvasWidth = Math.max(760, timelineScroll.clientWidth * zoom);
    timelineCanvas.style.width = `${canvasWidth}px`;
    renderRuler(spanNs);
    laneLabels.replaceChildren();
    const lanesContainer = byId("timeline-lanes");
    lanesContainer.replaceChildren();

    const lanes = new Map();
    events.forEach((event) => {
      if (!lanes.has(event.lane)) lanes.set(event.lane, []);
      lanes.get(event.lane).push(event);
    });
    [...lanes.entries()]
      .sort((left, right) => left[1][0].start_ns - right[1][0].start_ns)
      .forEach(([name, laneEvents]) => {
        const label = document.createElement("div");
        label.className = "lane-label";
        label.textContent = name;
        label.title = name;
        laneLabels.append(label);

        const lane = document.createElement("div");
        lane.className = "timeline-lane";
        laneEvents.forEach((event) => {
          const block = document.createElement("button");
          const key = eventKey(event);
          const description = eventDescription(event);
          block.type = "button";
          block.className = `event-block ${event.family}${event.return_code ? " event-error" : ""}`;
          block.dataset.eventKey = key;
          block.style.left = `${Number(event.start_ns) * 100 / spanNs}%`;
          block.style.width = `${Math.max(4, Number(event.duration_ns) * canvasWidth / spanNs)}px`;
          block.textContent = event.api;
          block.setAttribute("aria-label", description);
          block.setAttribute("aria-pressed", "false");
          block.addEventListener("pointerenter", () => showEventTooltip(block, event));
          block.addEventListener("pointerleave", () => hideEventTooltip(block));
          block.addEventListener("focus", () => showEventTooltip(block, event));
          block.addEventListener("blur", () => hideEventTooltip());
          block.addEventListener("click", (mouseEvent) => {
            mouseEvent.stopPropagation();
            selectEvent(event);
          });
          lane.append(block);
        });
        lanesContainer.append(lane);
      });

    const omitted = Number(latest.timeline.omitted || 0);
    const rangeParts = [
      formatNanoseconds(spanNs),
      `${formatCount(events.length)} visible`,
      `${formatCount(lanes.size)} function lanes`,
    ];
    if (omitted) rangeParts.push(`${formatCount(omitted)} invalid timestamps omitted`);
    setText("timeline-range", rangeParts.join(" | "));
    const empty = byId("timeline-empty");
    empty.textContent = latest.timeline.events.length
      ? "No timed CUDA events match the current filters."
      : latest.phase === "complete"
        ? "No timed CUDA events in this trace."
        : "Waiting for timed CUDA events.";
    empty.hidden = events.length > 0;
    byId("zoom-reset").textContent = `${Math.round(zoom * 100)}%`;
    requestAnimationFrame(() => {
      const selected = focusSelection
        ? [...timelineCanvas.querySelectorAll("[data-event-key]")]
            .find((node) => node.dataset.eventKey === selectedKey)
        : null;
      const targetCenter = selected
        ? selected.offsetLeft + selected.offsetWidth / 2
        : center * timelineScroll.scrollWidth;
      timelineScroll.scrollLeft = Math.max(0, targetCenter - timelineScroll.clientWidth / 2);
      laneLabels.style.transform = `translateY(${-timelineScroll.scrollTop}px)`;
    });
    updateSelectedStyles();
  }

  function renderEventsView() {
    const body = byId("recent-events");
    const events = visibleEvents().slice().reverse();
    body.replaceChildren();
    setText("event-count", `${formatCount(events.length)} visible`);
    if (!events.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 6;
      cell.className = "empty";
      cell.textContent = "No events match the current filters";
      row.append(cell);
      body.append(row);
      return;
    }
    events.forEach((event) => {
      const row = document.createElement("tr");
      const key = eventKey(event);
      row.dataset.eventKey = key;
      row.tabIndex = 0;
      row.setAttribute("aria-selected", "false");
      if (event.return_code) row.className = "event-error";
      row.title = `${event.file || "<unknown>"}:${event.line || 0}`;
      row.addEventListener("click", () => selectEvent(event));
      row.addEventListener("keydown", (keyEvent) => {
        if (keyEvent.key !== "Enter" && keyEvent.key !== " ") return;
        keyEvent.preventDefault();
        selectEvent(event);
      });
      [event.time, event.function || "<unknown>", event.api, event.kernel || "\u2014", event.duration]
        .forEach((value) => {
          const cell = document.createElement("td");
          cell.textContent = value;
          cell.title = String(value);
          row.append(cell);
        });
      const result = document.createElement("td");
      result.className = event.return_code ? "result-error" : "result-ok";
      result.textContent = event.return_code ? `ERR ${event.return_code}` : "OK";
      row.append(result);
      body.append(row);
    });
    updateSelectedStyles();
  }

  function reconcileSelection() {
    if (!selectedKey) return;
    const selected = latest.timeline.events.find((event) => eventKey(event) === selectedKey);
    if (selected) selectEvent(selected);
    else clearSelection();
  }

  function timelineSignature(data) {
    const events = data.timeline.events;
    const lastEvent = events.at(-1);
    return [
      data.generation,
      data.timeline.span_ns,
      data.timeline.omitted,
      events.length,
      lastEvent?.id || "",
    ].join(":");
  }

  function restoreEventFocus(key, surface) {
    if (!key || !surface) return;
    requestAnimationFrame(() => {
      const match = [...document.querySelectorAll("[data-event-key]")]
        .find((node) => node.dataset.eventKey === key
          && (surface === "timeline" ? node.matches(".event-block") : node.matches("tr")));
      match?.focus({preventScroll: true});
    });
  }

  function render(data) {
    const activeEventKey = document.activeElement?.dataset.eventKey;
    const activeSurface = document.activeElement?.matches(".event-block")
      ? "timeline"
      : document.activeElement?.matches("tr[data-event-key]")
        ? "table"
        : null;
    const signature = timelineSignature(data);
    const timelineChanged = signature !== renderedTimelineSignature;
    latest = data;
    refreshMs = Number(data.refresh_ms) || 200;
    setText("trace-title", data.trace_name || "Trace");
    setText("status-text", data.status);
    byId("status-badge").className = `status-badge ${statusClass(data.status)}`;
    const warning = byId("warning");
    const reasons = data.incomplete_reasons || [];
    const warningText = data.trace_error || data.summary_error
      || (reasons.length ? `Incomplete: ${reasons.join("; ")}` : "");
    warning.hidden = !warningText;
    warning.textContent = warningText || "";
    byId("token-form").hidden = true;

    setText("metric-events", formatCount(data.metrics.events));
    setText("metric-rate", `${Number(data.metrics.event_rate).toLocaleString(undefined, {maximumFractionDigits: 1, minimumFractionDigits: 1})} /s`);
    setText("metric-attributed", `${Number(data.metrics.attributed_percent).toFixed(1)}%`);
    setText("metric-errors", formatCount(data.metrics.cuda_errors));
    setText("metric-cpu", data.metrics.cpu_api);
    setText("metric-sync", data.metrics.synchronization);
    setText("metric-copied", data.metrics.copied);
    setText("metric-observed", data.metrics.observed);
    setText("metric-peak", data.metrics.peak);
    setText("metric-lost", formatCount(data.metrics.lost));
    setText("metric-dropped", formatCount(data.metrics.dropped));
    setText("metric-delivery-dropped", formatCount(data.metrics.delivery_dropped));
    setText("metric-malformed", formatCount(data.metrics.malformed));

    if (timelineChanged) {
      renderTimeline();
      renderEventsView();
      renderedTimelineSignature = signature;
      restoreEventFocus(activeEventKey, activeSurface);
    }
    reconcileSelection();
    renderRankList("top-apis", data.top_apis, "Waiting for CUDA events", "api");
    renderRankList("top-functions", data.top_functions, "No attributed functions yet", "function");
    renderRankList("top-kernels", data.top_kernels, "No resolved kernels yet", "kernel");
    renderMemory(data.memory_samples, data.metrics.peak);
    setText("updated-at", `Updated ${new Date().toLocaleTimeString()}`);
    setText("footer-refresh", `Refresh ${(refreshMs / 1000).toFixed(2)}s`);
  }

  function showConnectionError(error) {
    byId("status-badge").className = "status-badge error";
    setText("status-text", "DISCONNECTED");
    const warning = byId("warning");
    warning.hidden = false;
    warning.textContent = `Dashboard connection failed: ${error.message}`;
    byId("token-form").hidden = !error.needsToken;
    setText("updated-at", "Connection lost");
  }

  async function poll() {
    if (timer) window.clearTimeout(timer);
    try {
      if (!viewerToken) throw tokenError("This tab has no viewer token. Open the full private URL printed in the terminal, including #viewer_token=, or paste it below. Session storage must be enabled.");
      const response = await fetch("/api/state", {
        cache: "no-store",
        headers: {Authorization: `Bearer ${viewerToken}`},
      });
      if (response.status === 401) throw tokenError("The viewer token is not current. Paste the private URL printed by the running dashboard below.");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      render(await response.json());
    } catch (error) {
      showConnectionError(error);
    } finally {
      if (!paused) timer = window.setTimeout(poll, refreshMs);
    }
  }

  function setZoom(value) {
    zoom = Math.max(1, Math.min(16, value));
    renderTimeline(true);
  }

  document.querySelectorAll(".family-filter").forEach((button) => {
    button.addEventListener("click", () => {
      const family = button.dataset.family;
      if (activeFamilies.has(family)) activeFamilies.delete(family);
      else activeFamilies.add(family);
      const active = activeFamilies.has(family);
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      renderTimeline();
      renderEventsView();
      reconcileSelection();
    });
  });
  searchInput.addEventListener("input", () => {
    renderTimeline();
    renderEventsView();
    reconcileSelection();
  });
  byId("zoom-in").addEventListener("click", () => setZoom(zoom * 2));
  byId("zoom-out").addEventListener("click", () => setZoom(zoom / 2));
  byId("zoom-reset").addEventListener("click", () => setZoom(1));
  pauseButton.addEventListener("click", () => {
    paused = !paused;
    pauseButton.textContent = paused ? "Resume" : "Pause";
    pauseButton.setAttribute("aria-pressed", String(paused));
    if (paused && timer) window.clearTimeout(timer);
    if (!paused) poll();
  });
  byId("token-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = byId("token-input");
    const accepted = usePastedToken(input.value);
    input.value = "";
    if (accepted) poll();
    else showConnectionError(tokenError("That is not a viewer token or private dashboard URL, or session storage is disabled."));
  });
  timelineScroll.addEventListener("scroll", () => {
    laneLabels.style.transform = `translateY(${-timelineScroll.scrollTop}px)`;
    if (tooltipAnchor === document.activeElement) {
      requestAnimationFrame(positionEventTooltip);
    } else {
      hideEventTooltip();
    }
  });
  timelineScroll.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".event-block")) return;
    drag = {x: event.clientX, left: timelineScroll.scrollLeft};
    timelineScroll.setPointerCapture(event.pointerId);
    timelineScroll.style.cursor = "grabbing";
  });
  timelineScroll.addEventListener("pointermove", (event) => {
    if (!drag) return;
    timelineScroll.scrollLeft = drag.left - (event.clientX - drag.x);
  });
  const endDrag = () => {
    drag = null;
    timelineScroll.style.cursor = "";
  };
  timelineScroll.addEventListener("pointerup", endDrag);
  timelineScroll.addEventListener("pointercancel", endDrag);
  window.addEventListener("resize", () => {
    hideEventTooltip();
    if (latest) renderTimeline();
  });

  poll();
})();
"""


def _event_payload(event: _viewer.ViewerEvent) -> dict:
    return {
        "time": _viewer._time_cell(event.timestamp),
        "timestamp": event.timestamp,
        "pid": event.pid,
        "tid": event.tid,
        "function": event.function,
        "file": event.file,
        "line": event.line,
        "api": event.api,
        "kernel": event.kernel,
        "return_code": event.return_code,
        "duration_ns": event.duration_ns,
        "duration": _viewer._duration(event.duration_ns),
        "details": event.details,
        "span": event.span,
    }


def _timestamp_ns(timestamp: str) -> int | None:
    value = timestamp
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    utc = parsed.astimezone(datetime.timezone.utc)
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    delta = utc - epoch
    return (
        delta.days * 86_400_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
    )


def _api_family(api: str) -> str:
    if api.startswith("cuLaunch") or api == "cuGraphLaunch":
        return "launch"
    if api.startswith("cuMemcpy"):
        return "copy"
    if "Synchronize" in api:
        return "sync"
    if api.startswith("cuMem"):
        return "memory"
    return "other"


def _timeline_payload(model: _viewer.TraceModel) -> dict:
    recent = list(model.recent)
    selected = recent[-_TIMELINE_LIMIT:]
    first_sequence = model.events - len(recent) + (len(recent) - len(selected)) + 1
    stamped = []
    omitted = 0
    for offset, event in enumerate(selected):
        timestamp_ns = _timestamp_ns(event.timestamp)
        if timestamp_ns is None:
            omitted += 1
            continue
        stamped.append((first_sequence + offset, timestamp_ns, event))
    if not stamped:
        return {
            "origin_timestamp": None,
            "span_ns": 1,
            "omitted": omitted,
            "events": [],
        }

    origin_ns = min(timestamp_ns for _sequence, timestamp_ns, _event in stamped)
    origin_timestamp = min(
        stamped,
        key=lambda item: item[1],
    )[2].timestamp
    events = []
    span_ns = 1
    for sequence, timestamp_ns, event in stamped:
        start_ns = timestamp_ns - origin_ns
        item = _event_payload(event)
        item.update(
            {
                "id": str(sequence),
                "start_ns": start_ns,
                "family": _api_family(event.api),
                "lane": event.function or "<unknown>",
            }
        )
        events.append(item)
        span_ns = max(span_ns, start_ns + max(1, event.duration_ns))
    return {
        "origin_timestamp": origin_timestamp,
        "span_ns": span_ns,
        "omitted": omitted,
        "events": events,
    }


def _aggregate_payload(name: str, aggregate: _viewer.Aggregate) -> dict:
    return {
        "name": name,
        "count": aggregate.count,
        "total": _viewer._duration(aggregate.total_duration_ns),
        "max": _viewer._duration(aggregate.max_duration_ns),
    }


def _model_payload(
    model: _viewer.TraceModel,
    *,
    trace_name: str,
    waiting: bool,
    trace_error: str | None,
    summary_error: str | None,
    event_rate: float,
    refresh_seconds: float,
    generation: int = 0,
    delivery_dropped: int = 0,
) -> dict:
    attributed = 100.0 * model.attributed / model.events if model.events else 0.0
    top_apis = [
        _aggregate_payload(name, aggregate)
        for name, aggregate in _viewer._top(model.apis, _TOP_LIMIT)
    ]
    top_functions = []
    for (function, file_name, line), aggregate in _viewer._top(
        model.functions, _TOP_LIMIT
    ):
        item = _aggregate_payload(function, aggregate)
        item.update({"file": file_name, "line": line})
        top_functions.append(item)
    top_kernels = [
        _aggregate_payload(name, aggregate)
        for name, aggregate in _viewer._top(model.kernels, _TOP_LIMIT)
    ]
    recent_events = [
        _event_payload(event) for event in reversed(list(model.recent)[-_RECENT_LIMIT:])
    ]
    memory_samples = [
        {"timestamp": timestamp, "bytes": value}
        for timestamp, value in list(model.memory_samples)[-_MEMORY_LIMIT:]
    ]
    return {
        "schema_version": 1,
        "generation": generation,
        "trace_name": _viewer.sanitize_text(trace_name, 200),
        "status": _viewer.live_status(
            model,
            waiting=waiting,
            summary_error=summary_error,
            error=trace_error,
        ),
        "waiting": waiting,
        "trace_error": (
            _viewer.sanitize_text(trace_error) if trace_error is not None else None
        ),
        "summary_error": (
            _viewer.sanitize_text(summary_error) if summary_error is not None else None
        ),
        "refresh_ms": max(50, int(refresh_seconds * 1000)),
        "incomplete_reasons": model.incomplete_reasons(),
        "metrics": {
            "events": model.events,
            "event_rate": round(event_rate, 1),
            "attributed_percent": round(attributed, 1),
            "cuda_errors": model.cuda_errors,
            "cpu_api": _viewer._duration(model.total_api_duration_ns),
            "synchronization": _viewer._duration(model.synchronization_duration_ns),
            "copied": _viewer._bytes(model.successful_copy_bytes),
            "observed": _viewer._bytes(model.observed_outstanding_bytes),
            "peak": _viewer._bytes(model.observed_peak_bytes),
            "lost": model.summary_capture("lost_events"),
            "dropped": model.summary_capture("dropped_nested_calls"),
            "delivery_dropped": delivery_dropped,
            "malformed": model.malformed_lines,
        },
        "timeline": _timeline_payload(model),
        "top_apis": top_apis,
        "top_functions": top_functions,
        "top_kernels": top_kernels,
        "recent_events": recent_events,
        "memory_samples": memory_samples,
    }


class DashboardState:
    """Own live followers and expose immutable JSON-ready snapshots."""

    def __init__(
        self,
        trace: Path,
        summary: Path | None,
        recent_limit: int,
        refresh_seconds: float,
    ):
        self.trace = trace
        self.refresh_seconds = refresh_seconds
        self.follower = _follow.TraceFollower(trace, recent_limit=recent_limit)
        self.summaries = _follow.SummaryFollower(summary)
        self.rates = _tui.RateTracker()
        self.waiting = True
        self.following = False
        self.trace_error: str | None = None
        self.generation = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> None:
        with self._lock:
            try:
                update = self.follower.poll()
            except _viewer.ViewerError as exc:
                self.trace_error = str(exc)
                return
            self.trace_error = None
            self.waiting = update.waiting
            if self.waiting:
                self.following = False
            elif update.reset or not self.following:
                if update.reset:
                    self.generation += 1
                self.rates.reset()
                self.summaries.reset(self.follower.trace_mtime_ns)
                self.following = True
            if not self.waiting:
                self.summaries.poll(self.follower.model)
            self.rates.observe(time.monotonic(), self.follower.model.events)

    def payload(self) -> dict:
        with self._lock:
            return _model_payload(
                self.follower.model,
                trace_name=self.trace.name,
                waiting=self.waiting,
                trace_error=self.trace_error,
                summary_error=self.summaries.last_error,
                event_rate=self.rates.events_per_second,
                refresh_seconds=self.refresh_seconds,
                generation=self.generation,
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.refresh_seconds)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="metagross-web-follower",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.refresh_seconds * 2))
        with self._lock:
            self.follower.close()


class IngestConflict(Exception):
    """An ingest request does not match the active capture state."""


class IngestDashboardState:
    """Own one bounded in-memory capture received through authenticated POSTs."""

    def __init__(self, recent_limit: int, refresh_seconds: float):
        self.recent_limit = recent_limit
        self.refresh_seconds = refresh_seconds
        self.model = _viewer.TraceModel(recent_limit=recent_limit)
        self.rates = _tui.RateTracker()
        self.capture_id: str | None = None
        self.expected_sequence = 0
        self.terminal: str | None = None
        self.trace_name = "Waiting for Docker capture"
        self.delivery_dropped = 0
        self.generation = 0
        self.waiting = True
        self.trace_error: str | None = None
        self._lock = threading.Lock()

    def start_capture(self, capture_id: str, trace_name: str) -> dict:
        with self._lock:
            if capture_id == self.capture_id:
                return self._ack(status="live")
            if self.capture_id is not None:
                self.generation += 1
            self.model = _viewer.TraceModel(recent_limit=self.recent_limit)
            self.rates.reset()
            self.capture_id = capture_id
            self.expected_sequence = 0
            self.terminal = None
            self.trace_name = _viewer.sanitize_text(trace_name, 200)
            self.delivery_dropped = 0
            self.waiting = False
            self.trace_error = None
            self.rates.observe(time.monotonic(), 0)
            return self._ack(status="live")

    def ingest_events(
        self,
        capture_id: str,
        sequence: int,
        events: list[_viewer.ViewerEvent],
    ) -> dict:
        with self._lock:
            self._require_active(capture_id)
            if self.terminal is not None:
                raise IngestConflict("capture is already terminal")
            if sequence > self.expected_sequence:
                raise IngestConflict("capture sequence has a gap")
            if sequence == self.expected_sequence:
                for event in events:
                    self.model.observe(event)
                self.expected_sequence += 1
                self.rates.observe(time.monotonic(), self.model.events)
            return self._ack(next_sequence=self.expected_sequence)

    def finish_capture(
        self,
        capture_id: str,
        sequence: int,
        summary: dict,
    ) -> dict:
        with self._lock:
            self._require_active(capture_id)
            if self.terminal == "finished":
                if sequence != self.expected_sequence:
                    raise IngestConflict("capture finish sequence does not match")
                return self._ack(
                    status="finished",
                    next_sequence=self.expected_sequence,
                )
            if self.terminal is not None:
                raise IngestConflict("capture is already terminal")
            if sequence != self.expected_sequence:
                raise IngestConflict("capture finish sequence does not match")
            self.model.load_summary(summary)
            self.delivery_dropped = self.model.summary_capture("delivery_dropped")
            self.terminal = "finished"
            self.rates.observe(time.monotonic(), self.model.events)
            return self._ack(
                status="finished",
                next_sequence=self.expected_sequence,
            )

    def abort_capture(self, capture_id: str, message: str) -> dict:
        with self._lock:
            self._require_active(capture_id)
            if self.terminal == "aborted":
                return self._ack(status="aborted")
            if self.terminal is not None:
                raise IngestConflict("capture is already terminal")
            self.terminal = "aborted"
            self.trace_error = _viewer.sanitize_text(message)
            self.waiting = False
            return self._ack(status="aborted")

    def payload(self) -> dict:
        with self._lock:
            self.rates.observe(time.monotonic(), self.model.events)
            return _model_payload(
                self.model,
                trace_name=self.trace_name,
                waiting=self.waiting,
                trace_error=self.trace_error,
                summary_error=None,
                event_rate=self.rates.events_per_second,
                refresh_seconds=self.refresh_seconds,
                generation=self.generation,
                delivery_dropped=self.delivery_dropped,
            )

    @staticmethod
    def start() -> None:
        return

    @staticmethod
    def close() -> None:
        return

    def _require_active(self, capture_id: str) -> None:
        if self.capture_id is None or capture_id != self.capture_id:
            raise IngestConflict("capture is not active")

    def _ack(
        self,
        *,
        status: str | None = None,
        next_sequence: int | None = None,
    ) -> dict:
        response = {
            "schema_version": 1,
            "capture_id": self.capture_id,
        }
        if status is not None:
            response["status"] = status
        if next_sequence is not None:
            response["next_sequence"] = next_sequence
        return response


def _client_is_loopback(address: tuple) -> bool:
    try:
        return ipaddress.ip_address(address[0]).is_loopback
    except (ValueError, IndexError, TypeError):
        return False


def _client_is_internal(address: tuple) -> bool:
    # Internal means not routable on the public internet: loopback, private,
    # link-local, and shared (CGNAT, e.g. Tailscale) ranges all qualify.
    try:
        return not ipaddress.ip_address(address[0]).is_global
    except (ValueError, IndexError, TypeError):
        return False


def _is_ipv4_literal(name: str) -> bool:
    try:
        ipaddress.IPv4Address(name)
    except ValueError:
        return False
    return True


_MAX_CONNECTIONS = 32
_SOCKET_TIMEOUT_SECONDS = 10


class _DashboardServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        self.viewer_token = secrets.token_urlsafe(32)
        self._slots = threading.BoundedSemaphore(_MAX_CONNECTIONS)
        super().__init__(*args, **kwargs)
        # A non-loopback bind (the --host opt-in) lets LAN viewers read state.
        self.lan_mode = not ipaddress.ip_address(self.server_address[0]).is_loopback

    def process_request(self, request, client_address) -> None:
        # One thread per connection: without a cap, idle clients could hold
        # an unbounded number of them.
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()  # no thread was started to release it
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request, client_address) -> None:
        # A viewer that closes its connection mid-response is routine. The
        # default would print a traceback, which under `--web` lands in the
        # terminal that carries the trace table.
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


class _IngestRequestError(Exception):
    def __init__(self, status: int, message: str, *, authenticate: bool = False):
        super().__init__(message)
        self.status = status
        self.message = message
        self.authenticate = authenticate


class _DashboardRequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MetagrossDashboard/1"
    timeout = _SOCKET_TIMEOUT_SECONDS  # drop a client that goes quiet

    def log_message(self, _format: str, *_args) -> None:
        return

    def _valid_host(self, *, allow_lan: bool = True) -> bool:
        values = self.headers.get_all("Host", [])
        if len(values) != 1:
            return False
        host = values[0].lower()
        if host in ("127.0.0.1", "localhost"):
            return True
        name, separator, port = host.rpartition(":")
        if (
            separator
            and name in ("127.0.0.1", "localhost")
            and port == str(self.server.server_port)
        ):
            return True
        if not (allow_lan and getattr(self.server, "lan_mode", False)):
            return False
        # LAN opt-in: also accept a numeric IPv4 Host. DNS names stay rejected,
        # which is what defeats DNS rebinding.
        if not separator:
            return _is_ipv4_literal(host)
        return _is_ipv4_literal(name) and port == str(self.server.server_port)

    def _send(
        self,
        body: bytes,
        content_type: str,
        *,
        status: int = http.server.HTTPStatus.OK,
        headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; img-src 'self'; base-uri 'none'; "
            "form-action 'none'; frame-ancestors 'none'",
        )
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, value: dict) -> None:
        body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._send(body, "application/json; charset=utf-8")

    def _send_ack(self, value: dict) -> None:
        body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(body) > _MAX_RESPONSE_BODY:
            self._send_error(
                http.server.HTTPStatus.INTERNAL_SERVER_ERROR,
                "response is too large",
            )
            return
        self._send(body, "application/json; charset=utf-8")

    def _send_error(
        self,
        status: int,
        _message: str,
        *,
        authenticate: bool = False,
    ) -> None:
        public_messages = {
            400: "invalid request",
            401: "unauthorized",
            403: "forbidden",
            404: "not found",
            405: "method not allowed",
            409: "conflict",
            411: "length required",
            413: "request too large",
            415: "unsupported media type",
            500: "internal server error",
        }
        body = json.dumps(
            {"error": public_messages.get(int(status), "request failed")},
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        headers = ((("WWW-Authenticate", "Bearer"),) if authenticate else ())
        self._send(
            body,
            "application/json; charset=utf-8",
            status=status,
            headers=headers,
        )

    def _serve(self) -> None:
        # Defense in depth for the --host opt-in: never serve a client whose
        # address is on the public internet, even if a port is forwarded.
        if not _client_is_internal(self.client_address):
            self.close_connection = True
            self._send(
                b"forbidden\n",
                "text/plain; charset=utf-8",
                status=http.server.HTTPStatus.FORBIDDEN,
            )
            return
        if not self._valid_host():
            self._send(
                b"invalid Host header\n",
                "text/plain; charset=utf-8",
                status=http.server.HTTPStatus.FORBIDDEN,
            )
            return
        path = urllib.parse.urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send(_INDEX_HTML, "text/html; charset=utf-8")
        elif path == "/app.css":
            self._send(_APP_CSS, "text/css; charset=utf-8")
        elif path == "/app.js":
            self._send(_APP_JS, "text/javascript; charset=utf-8")
        elif path == "/api/state":
            if not self._authorized(self.server.viewer_token):
                self._send_error(
                    http.server.HTTPStatus.UNAUTHORIZED,
                    "invalid bearer token",
                    authenticate=True,
                )
                return
            self._send_json(self.server.dashboard_state.payload())
        elif path == "/logo.svg":
            self._send(_LOGO_SVG, "image/svg+xml; charset=utf-8")
        elif path == "/favicon.svg":
            self._send(_FAVICON_SVG, "image/svg+xml; charset=utf-8")
        else:
            self._send(
                b"not found\n",
                "text/plain; charset=utf-8",
                status=http.server.HTTPStatus.NOT_FOUND,
            )

    def _authorized(self, token: str | None) -> bool:
        authorization = self.headers.get_all("Authorization", [])
        if token is None or len(authorization) != 1:
            return False
        try:
            return hmac.compare_digest(
                authorization[0].encode("latin-1"),
                f"Bearer {token}".encode("ascii"),
            )
        except UnicodeEncodeError:
            return False

    def _post_preflight(self) -> tuple[str, int]:
        # Capture ingest stays loopback-only even when --host admits LAN
        # viewers: the privileged producer only ever connects from 127.0.0.1.
        if not _client_is_loopback(self.client_address):
            raise _IngestRequestError(
                http.server.HTTPStatus.FORBIDDEN,
                "capture ingest is loopback-only",
            )
        if not self._valid_host(allow_lan=False):
            raise _IngestRequestError(
                http.server.HTTPStatus.FORBIDDEN,
                "invalid Host header",
            )
        path = urllib.parse.urlsplit(self.path).path
        limits = {
            "/api/capture/start": _MAX_CONTROL_BODY,
            "/api/capture/events": _MAX_EVENT_BODY,
            "/api/capture/finish": _MAX_FINISH_BODY,
            "/api/capture/abort": _MAX_CONTROL_BODY,
        }
        if not isinstance(self.server.dashboard_state, IngestDashboardState):
            raise _IngestRequestError(
                http.server.HTTPStatus.NOT_FOUND,
                "not found",
            )
        if path not in limits:
            raise _IngestRequestError(
                http.server.HTTPStatus.NOT_FOUND,
                "not found",
            )

        if not self._authorized(self.server.ingest_token):
            raise _IngestRequestError(
                http.server.HTTPStatus.UNAUTHORIZED,
                "invalid bearer token",
                authenticate=True,
            )
        if self.headers.get_all("Transfer-Encoding", []):
            raise _IngestRequestError(
                http.server.HTTPStatus.BAD_REQUEST,
                "Transfer-Encoding is not supported",
            )
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            raise _IngestRequestError(
                http.server.HTTPStatus.LENGTH_REQUIRED,
                "Content-Length is required",
            )
        if (
            len(lengths) != 1
            or not lengths[0]
            or any(character < "0" or character > "9" for character in lengths[0])
        ):
            raise _IngestRequestError(
                http.server.HTTPStatus.BAD_REQUEST,
                "Content-Length must be one non-negative integer",
            )
        if len(lengths[0]) > 20:
            raise _IngestRequestError(
                http.server.HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request body is too large",
            )
        length = int(lengths[0])
        if length > limits[path]:
            raise _IngestRequestError(
                http.server.HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request body is too large",
            )
        content_types = self.headers.get_all("Content-Type", [])
        if len(content_types) != 1 or not self._valid_json_type(content_types[0]):
            raise _IngestRequestError(
                http.server.HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Content-Type must be application/json",
            )
        return path, length

    @staticmethod
    def _valid_json_type(value: str) -> bool:
        parts = [part.strip().lower() for part in value.split(";")]
        if not parts or parts[0] != "application/json" or len(parts) > 2:
            return False
        if len(parts) == 1:
            return True
        return parts[1] in ("charset=utf-8", 'charset="utf-8"')

    def _read_json(self, length: int):
        try:
            body = self.rfile.read(length)
        except OSError as exc:
            raise _IngestRequestError(
                http.server.HTTPStatus.BAD_REQUEST,
                f"cannot read request body: {exc}",
            ) from None
        if len(body) != length:
            raise _IngestRequestError(
                http.server.HTTPStatus.BAD_REQUEST,
                "request body ended early",
            )
        try:
            return _viewer._load_bounded_json(body)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise _IngestRequestError(
                http.server.HTTPStatus.BAD_REQUEST,
                f"invalid JSON body: {exc}",
            ) from None

    @staticmethod
    def _envelope(value, fields: set[str]) -> dict:
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        expected = fields | {"schema_version"}
        if set(value) != expected:
            raise ValueError("request body has unexpected or missing fields")
        version = value.get("schema_version")
        if not _viewer._is_int(version) or version != 1:
            raise ValueError("schema_version must be 1")
        return value

    @staticmethod
    def _capture_id(value) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 32
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("capture_id must be 32 lowercase hexadecimal characters")
        return value

    @staticmethod
    def _sequence(value) -> int:
        if not _viewer._is_int(value) or value < 0:
            raise ValueError("sequence must be a non-negative integer")
        return value

    def _dispatch_post(self, path: str, body) -> dict:
        state = self.server.dashboard_state
        if path == "/api/capture/start":
            request = self._envelope(body, {"capture_id", "trace_name"})
            capture_id = self._capture_id(request["capture_id"])
            trace_name = request["trace_name"]
            if (
                not isinstance(trace_name, str)
                or not trace_name
                or len(trace_name) > 255
                or Path(trace_name).name != trace_name
            ):
                raise ValueError("trace_name must be a non-empty basename")
            return state.start_capture(capture_id, trace_name)

        if path == "/api/capture/events":
            request = self._envelope(
                body,
                {"capture_id", "sequence", "events"},
            )
            capture_id = self._capture_id(request["capture_id"])
            sequence = self._sequence(request["sequence"])
            records = request["events"]
            if (
                not isinstance(records, list)
                or not records
                or len(records) > _MAX_BATCH_EVENTS
            ):
                raise ValueError("events must contain between 1 and 128 records")
            events = [_viewer.parse_event(record) for record in records]
            return state.ingest_events(capture_id, sequence, events)

        if path == "/api/capture/finish":
            request = self._envelope(
                body,
                {"capture_id", "sequence", "summary"},
            )
            capture_id = self._capture_id(request["capture_id"])
            sequence = self._sequence(request["sequence"])
            summary = request["summary"]
            validation_model = _viewer.TraceModel(recent_limit=1)
            validation_model.load_summary(summary)
            return state.finish_capture(capture_id, sequence, summary)

        request = self._envelope(body, {"capture_id", "message"})
        capture_id = self._capture_id(request["capture_id"])
        message = request["message"]
        if not isinstance(message, str) or not message:
            raise ValueError("message must be a non-empty string")
        return state.abort_capture(capture_id, message)

    def handle_expect_100(self) -> bool:
        try:
            self._post_preflight()
        except _IngestRequestError as exc:
            self.close_connection = True
            self._send_error(
                exc.status,
                exc.message,
                authenticate=exc.authenticate,
            )
            return False
        self.send_response_only(http.server.HTTPStatus.CONTINUE)
        self.end_headers()
        return True

    def do_GET(self) -> None:
        self._serve()

    def do_HEAD(self) -> None:
        self._serve()

    def do_OPTIONS(self) -> None:
        if not _client_is_internal(self.client_address):
            self.close_connection = True
            self._send_error(http.server.HTTPStatus.FORBIDDEN, "forbidden")
            return
        if not self._valid_host():
            self._send_error(
                http.server.HTTPStatus.FORBIDDEN,
                "invalid Host header",
            )
            return
        self._send_error(
            http.server.HTTPStatus.METHOD_NOT_ALLOWED,
            "method not allowed",
        )

    def do_POST(self) -> None:
        try:
            path, length = self._post_preflight()
            body = self._read_json(length)
            response = self._dispatch_post(path, body)
        except _IngestRequestError as exc:
            self.close_connection = True
            self._send_error(
                exc.status,
                exc.message,
                authenticate=exc.authenticate,
            )
            return
        except IngestConflict as exc:
            self._send_error(http.server.HTTPStatus.CONFLICT, str(exc))
            return
        except (_viewer.ViewerError, ValueError) as exc:
            self._send_error(http.server.HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_ack(response)


def run_web_dashboard(
    trace: Path | None,
    summary: Path | None,
    recent_limit: int,
    refresh_seconds: float,
    port: int = _DEFAULT_PORT,
    *,
    ingest_token: str | None = None,
    host: str = _DEFAULT_HOST,
    on_ready=None,
) -> int:
    """Serve a file-backed or ingest-backed dashboard.

    The server binds loopback unless host opts in to a LAN address. Capture
    ingest stays loopback-only either way. `on_ready`, when given, receives
    the bound port once the socket is listening and the URL is printed.
    """
    if trace is None:
        if ingest_token is None:
            print(
                "metagross view: receive mode requires an ingest token",
                file=sys.stderr,
            )
            return 1
        state = IngestDashboardState(recent_limit, refresh_seconds)
    else:
        state = DashboardState(trace, summary, recent_limit, refresh_seconds)
    try:
        server = _DashboardServer((host, port), _DashboardRequestHandler)
    except OSError as exc:
        print(f"metagross view: cannot start web dashboard: {exc}", file=sys.stderr)
        state.close()
        return 1
    server.dashboard_state = state
    server.ingest_token = ingest_token
    state.start()
    if server.lan_mode:
        print(
            "metagross view: warning: the dashboard is reachable from the "
            f"network on {host}:{server.server_port}. It uses plain HTTP, so "
            "the viewer token and trace data cross the network unencrypted, "
            "and anyone with the URL can read trace source paths and timing. "
            "Clients with public-internet addresses are refused and capture "
            "ingest stays loopback-only; use this only on a trusted internal "
            "network.",
            file=sys.stderr,
        )
    display_host = "<this-host-ip>" if host == "0.0.0.0" else host
    address = (
        f"http://{display_host}:{server.server_port}/"
        f"#viewer_token={server.viewer_token}"
    )
    print(f"metagross view: dashboard available at {address}", file=sys.stderr)
    if host == "0.0.0.0":
        print(
            "metagross view: replace <this-host-ip> with an IPv4 address of "
            "this machine; hostnames are rejected",
            file=sys.stderr,
        )
    try:
        if on_ready is not None:
            on_ready(server.server_port)
        server.serve_forever(poll_interval=min(0.2, refresh_seconds))
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
        state.close()
    return 0
