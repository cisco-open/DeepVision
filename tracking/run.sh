while getopts i:o:c:r: flag
do
    case "${flag}" in
        i) input_stream=${OPTARG};;
        o) output_stream=${OPTARG};;
        c) classId=${OPTARG};;
        r) redis=${OPTARG};;
    esac
done
docker stop mmtracking
docker build -t mmtracking:latest .
docker network create redisconnection
docker run --name mmtracking --net redisconnection --gpus all -itd --rm mmtracking:latest
if [ ! "$(docker ps -a -q -f name='redis')" ]; then
  docker run  -p 6379:6379 --name redis --net redisconnection -d --rm redis
fi
docker exec -it mmtracking bash -c "python tracker.py mmtracking/configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py --input_stream ${input_stream} --output_stream ${output_stream} --classId ${classId} --redis ${redis}"
docker cp mmtracking:/mmtracking/mot.mp4 .

#configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py --input_stream camera:0 --output_stream camera:0:mot --classId PERSON --device  --redis redis://redis:6379"
#For example:
#bash -x run.sh -i camera:0 -o camera:0:mot -c PERSON -r redis://redis:6379