# Ethosight benchmarking

### Instruction for running

1. Please call `git submodule update --recursive` or `git submodule update --init --recursive` depending on git version  
2. Add your embeddings file in `./ethosight/embeddings/` folder  
3. Run deepvison with ethosight profile `./run ethosight up --build`
4. Run benchmarking script, here is an example `bash run_ethosight_benchmark.sh "data/liverpool.mp4,10.00;data/basketball.mp4,30.00" "general.embeddings"`  
here "data/liverpool.mp4,10.00;data/basketball.mp4,30.00" is the first parameter where you provide list of videos and their output fps that you desire, if you don't want downsampling you can put the same fps as original video. The second parameter "general.embeddings" is the embedding files name, just put the name and it will look inside ./ethosight/embeddings/ folder  

Everything in benchmarking script will work async way, so each video will process asynchronously. You will see the resulting output files in ./benchmarks/*.json files for each video. And vor each video will be created the following redis streams "camera:{video_filename}" as frames producing stream and "camera:{video_filename}:affscores" as affinity scores results.  

