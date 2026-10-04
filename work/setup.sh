#!/bin/bash
# setup.sh - Installer for Samantha Assistant on openEuler

set -e  # exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER_SCRIPT="$PROJECT_DIR/samantha.sh"

echo "[1/3] Checking Python3 and pip3..."

# ensure Python3 and pip3 are installed
if ! command -v python3 >/dev/null 2>&1; then
    echo "Python3 not found. Installing..."
    dnf install -y python3
fi

if ! command -v pip3 >/dev/null 2>&1; then
    echo "pip3 not found. Installing..."
    dnf install -y python3-pip
fi

# Install dependency packages
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Samantha with LangGraph requires Python 3.10 or newer."
    return 1 2>/dev/null || exit 1
fi

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip3 install -r "$PROJECT_DIR/requirements.txt"
else
    echo "No requirements.txt found, skipping..."
fi

echo "[2/3] Registering the Samantha shell function..."
SOURCE_LINE="source \"$(realpath "$WRAPPER_SCRIPT")\""
if ! grep -Fxq -- "$SOURCE_LINE" ~/.bashrc 2>/dev/null; then
    echo "$SOURCE_LINE" >> ~/.bashrc
fi
source "$WRAPPER_SCRIPT"

echo "[3/3] Setup complete."
echo "Samantha is ready to help"
echo "For example, in your terminal, you can type \"samantha create a file named test.txt\" instead of \"touch test.txt\""
