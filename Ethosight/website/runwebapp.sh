#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

python3 "$SCRIPT_DIR/manage.py" makemigrations
python3 "$SCRIPT_DIR/manage.py" migrate
python3 "$SCRIPT_DIR/manage.py" createsuperuser
python3 "$SCRIPT_DIR/manage.py" runserver 8080
