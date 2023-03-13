docker stop producer
docker build --no-cache -t producer:latest .
docker network create redisconnection
docker run -p 6379:6379 --name redis --net redisconnection -d --rm redis
docker run -p 5002:5001 --name producer --net redisconnection -v $(pwd)/:/app -itd --rm producer:latest
docker exec -it producer bash -c "python3 producer.py --verbose True -u redis://redis:6379 data/race.mp4 & python3 server.py -u redis://redis:6379"
