SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

cp "$SCRIPT_DIR/../config/nginx/nginx.conf" /tmp/nginx.conf
while true ; do
	docker rm -f ethosight-nginx
	docker run --name ethosight-nginx --network host -v /tmp/nginx.conf:/etc/nginx/nginx.conf:ro -p 80:80 -t nginx
	sleep 5 
done
