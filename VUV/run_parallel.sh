#!/bin/bash
# Parallel WORLD processing launcher

# Set environment variable to fix libstdc++ version issue
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run Python script
python3 parallel_process.py "$@"