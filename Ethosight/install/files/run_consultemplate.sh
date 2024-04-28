#!/bin/bash

# Load environment variables, if any
source /etc/environment

# Define paths
TEMPLATE_PATH="/etc/consul-template.d/nginx.template"  # Path where nginx.template is stored
CONFIG_PATH="/tmp/nginx/nginx.conf"  # Output path for the rendered configuration

# Logging setup
LOG_FILE="/var/log/consul-template.log"

# Logging the start
echo "$(date): Starting consul-template..." >> $LOG_FILE

# Run consul-template
consul-template -template "${TEMPLATE_PATH}:${CONFIG_PATH}:docker exec ethosight-nginx nginx -s reload" \
|| echo "$(date): Error during consul-template execution." >> $LOG_FILE

# Logging completion
echo "$(date): consul-template processing complete." >> $LOG_FILE
