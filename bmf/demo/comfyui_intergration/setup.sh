#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 BabitMF
set -e

# This script automates the setup process for the BMF and ComfyUI integration.

# 1. Ensure Python version is compatible with ComfyUI (> 3.11) and set up environment
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python not found in PATH." >&2
    exit 1
fi

CURRENT_PYTHON_VERSION="$($PYTHON_BIN -V 2>&1 | awk '{print $2}')"
CURRENT_MAJOR="$(echo "$CURRENT_PYTHON_VERSION" | cut -d. -f1)"
CURRENT_MINOR_RAW="$(echo "$CURRENT_PYTHON_VERSION" | cut -d. -f2)"
CURRENT_MINOR="$(echo "$CURRENT_MINOR_RAW" | sed 's/[^0-9].*$//')"
CURRENT_MINOR="${CURRENT_MINOR:-0}"

# ComfyUI compatibility requires Python > 3.11 (i.e., Python 3.12+)
if [ "$CURRENT_MAJOR" -lt 3 ] || { [ "$CURRENT_MAJOR" -eq 3 ] && [ "$CURRENT_MINOR" -le 11 ]; }; then
    echo "Error: Found Python $CURRENT_PYTHON_VERSION at $PYTHON_BIN, but ComfyUI compatibility requires Python > 3.11." >&2
    echo "Please switch your Python interpreter (e.g., via pyenv) to Python 3.11+ and re-run." >&2
    exit 1
fi

echo "Using Python: $PYTHON_BIN ($CURRENT_PYTHON_VERSION)"

# Ensure pip is run via the selected Python interpreter
PIP_CMD="$PYTHON_BIN -m pip"

# Set up ComfyUI integration environment
INTEGRATION_DIR="."
cd "$INTEGRATION_DIR"

# 2. Clone ComfyUI if not already present
if [ ! -d "ComfyUI" ]; then
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    git switch --detach f228367c5e3906de194968fa9b6fbe7aa9987bfa
    cd ..
    #git clone https://github.com/ischencheng/ComfyUI.git
else
    echo "ComfyUI directory already exists. Skipping clone."
fi

# 3. Install dependencies
echo "Installing dependencies for ComfyUI with $PYTHON_BIN..."
if [ -f "ComfyUI/requirements.txt" ]; then
    $PIP_CMD install -r ComfyUI/requirements.txt
else
     echo "Warning: ComfyUI/requirements.txt not found."
fi

echo "Setup complete!"
echo "You can now run the integration using: $PYTHON_BIN run_bmf_comfy.py"
