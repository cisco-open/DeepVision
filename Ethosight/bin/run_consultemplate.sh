#consul-template -template "nginx.template:nginx.conf"
consul-template -template "nginx.template:/tmp/nginx.conf:docker exec ethosight-nginx nginx -s reload" 
