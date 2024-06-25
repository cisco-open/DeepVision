#!/usr/bin/bash
LOG_FILE="/var/log/consul-template.log"
TEMPLATE_PATH="/etc/consul-template.d/nginx.template"
CONFIG_PATH="/tmp/nginx/nginx.conf"

# Check and load Ethosight specific environment variables
if [ -f /etc/ethosight_environment ]; then
    source /etc/ethosight_environment
else
    echo "$(date): Ethosight environment file not found, skipping load." >> $LOG_FILE
fi

# Logging the start
echo "$(date): Starting consul-template..." >> $LOG_FILE

# Run consul-template
consul-template -template "${TEMPLATE_PATH}:${CONFIG_PATH}:docker exec ethosight-nginx nginx -s reload" \
|| echo "$(date): Error during consul-template execution." >> $LOG_FILE

# Logging completion
echo "$(date): consul-template processing complete." >> $LOG_FILE
