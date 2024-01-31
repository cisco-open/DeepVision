#consul-template -template "nginx.template:nginx.conf"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
consul-template -template "$SCRIPT_DIR/../config/consul/nginx.template:/tmp/nginx.conf:docker exec ethosight-nginx nginx -s reload"
