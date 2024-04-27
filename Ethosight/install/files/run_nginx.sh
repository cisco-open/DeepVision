#!/bin/bash

CONFIG_DIR="/tmp/nginx"
NGINX_CONF="$CONFIG_DIR/nginx.conf"

# Function to restart Nginx Docker container
restart_nginx() {
    echo "Restarting Nginx container due to configuration change..."
    docker rm -f ethosight-nginx
    docker run --name ethosight-nginx --network host -v "$CONFIG_DIR":/etc/nginx:ro -p 80:80 -d nginx
}

# Initial start of the Nginx container
restart_nginx

# Monitor nginx.conf for changes
while inotifywait -e modify,move_self,create,delete $NGINX_CONF; do
    restart_nginx
done
