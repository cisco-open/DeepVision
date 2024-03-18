SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

docker run --name tmp-nginx -d nginx
docker cp tmp-nginx:/etc/nginx /tmp
docker stop tmp-nginx
docker rm tmp-nginx

cp "$SCRIPT_DIR/../config/nginx/nginx.conf" /tmp/nginx/nginx.conf

while true ; do
    docker rm -f ethosight-nginx

    docker run --name ethosight-nginx --network host \
        -v /tmp/nginx:/etc/nginx:ro \
        -p 80:80 -t nginx

    sleep 5 
done

