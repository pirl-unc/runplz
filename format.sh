#!/usr/bin/env bash

set -e

SOURCES="runplz tests"

echo "Running ruff format..."
ruff format $SOURCES

echo "Formatting complete!"
