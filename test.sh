#!/usr/bin/env bash
set -e
pytest --cov=runplz/ --cov-report=term-missing tests
