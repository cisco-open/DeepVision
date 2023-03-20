docker stop producer
docker build --no-cache -t producer:latest .
docker network create redisconnection
docker run -p 6379:6379 --name redis_vision --net redisconnection -d --rm redis_vision
docker run -p 5002:5001 --name producer --net redisconnection -v $(pwd)/:/app -itd --rm producer:latest
docker exec -it producer bash -c "python3 producer.py --verbose True -u redis://redis_vision:6379 data/race.mp4 & python3 server.py -u redis://redis_vision:6379"
