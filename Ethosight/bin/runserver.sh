#!/bin/bash

COMMAND=$1
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
python3 "$SCRIPT_DIR/runserver.py" $COMMAND
