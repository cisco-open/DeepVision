### Building docker image
``docker build -t ethosight .``
### Running the image container
``docker run -itd  --gpus all --name ethosight --rm ethosight:latest``

### Executing specific script on the container
``docker exec -it -e OPENAI_API_KEY ethosight /bin/bash -c "source /miniconda/etc/profile.d/conda.sh && conda activate imagebind && python <script_name>.py"``