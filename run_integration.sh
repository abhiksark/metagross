#!/bin/sh
# run_integration.sh
# Runs the root-gated live integration suite for Metagross.
cd "$(dirname "$0")"
exec sudo env RUN_EBPF_INTEGRATION=1 /usr/bin/python3 -m unittest -v tests.test_metagross.LiveTraceTest
