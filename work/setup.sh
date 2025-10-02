#!/bin/bash
# setup.sh - Installer for Samantha Assistant on openEuler

set -e  # exit on error

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)" # get the directory of the script
WRAPPER_SCRIPT="$PROJECT_DIR/samantha.sh"
INSTALL_PATH="/usr/local/bin/samantha"

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
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip3 install -r "$PROJECT_DIR/requirements.txt"
else
    echo "No requirements.txt found, skipping..."
fi

# echo "[2/3] Making samantha.sh executable..."
# chmod +x "$WRAPPER_SCRIPT"

# echo "[3/3] Linking samantha to /usr/local/bin..."
# ln -sf "$WRAPPER_SCRIPT" "$INSTALL_PATH"

echo "source $(realpath "$WRAPPER_SCRIPT")" >> ~/.bashrc
source ~/.bashrc

echo "Samantha is ready to help"
echo "For example, in your terminal, you can type \"samantha create a file named test.txt\" instead of \"touch test.txt\""