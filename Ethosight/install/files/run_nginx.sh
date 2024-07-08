#!/bin/bash

CONFIG_DIR="/tmp/nginx"
NGINX_CONF="$CONFIG_DIR/nginx.conf"
CONTAINER_NAME="ethosight-nginx"
LOG_DIR="$HOME/nginx_logs"
LOG_FILE="$LOG_DIR/nginx.log"

# Ensure the log directory and file exist
mkdir -p $LOG_DIR || { echo "Failed to create log directory: $LOG_DIR"; exit 1; }
touch $LOG_FILE || { echo "Failed to create log file: $LOG_FILE"; exit 1; }

# Log initial status
echo "Running as user: $(whoami)" >> $LOG_FILE
echo "Current directory: $(pwd)" >> $LOG_FILE
echo "Starting script run_nginx.sh" >> $LOG_FILE

# Ensure the configuration directory exists and has the necessary permissions
#mkdir -p $CONFIG_DIR
#chmod 755 $CONFIG_DIR

# Function to restart Nginx Docker container
restart_nginx() {
    echo "Restarting Nginx container due to configuration change..." >> $LOG_FILE
    docker rm -f $CONTAINER_NAME &>> $LOG_FILE
    docker run --name $CONTAINER_NAME --network host -v "$NGINX_CONF":/etc/nginx/nginx.conf:ro -p 80:80 -d nginx &>> $LOG_FILE
    if [ $? -ne 0 ]; then
        echo "Failed to start Docker container." >> $LOG_FILE
    fi
}

# Initial start of the Nginx container
restart_nginx

# Monitor nginx.conf for changes
while inotifywait -e modify,move_self,create,delete $NGINX_CONF; do
    restart_nginx
done
