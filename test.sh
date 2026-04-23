#!/usr/bin/env bash
set -e
python -m pytest --cov=runplz --cov-report=term-missing tests
