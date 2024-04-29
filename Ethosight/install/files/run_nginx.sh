#!/bin/bash

CONFIG_DIR="/tmp/nginx"
NGINX_CONF="$CONFIG_DIR/nginx.conf"
CONTAINER_NAME="ethosight-nginx"

# Ensure the configuration directory exists and has the necessary permissions
mkdir -p $CONFIG_DIR
chmod 755 $CONFIG_DIR

# Function to restart Nginx Docker container
restart_nginx() {
    echo "Restarting Nginx container due to configuration change..."
    # Stop and remove the current Nginx container if it exists
    docker rm -f $CONTAINER_NAME
    # Start a new Nginx container with the custom nginx.conf mounted
    docker run --name $CONTAINER_NAME --network host -v "$NGINX_CONF":/etc/nginx/nginx.conf:ro -p 80:80 -d nginx
}

# Initial start of the Nginx container
restart_nginx

# Monitor nginx.conf for changes
while inotifywait -e modify,move_self,create,delete $NGINX_CONF; do
    restart_nginx
done

