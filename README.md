# CiscoDeepVision

## GETTING STARTED:

### Cloning the repo

While cloning the repo use the command:
```bash
git clone --recurse-submodules https://github.com/CiscoDeepVision/CiscoDeepVision.git
```

### Running  and building

Launch docker compose

 ```bash
    docker-compose up --build
```

### For grafana dashboard and Webserver  

To see grafana dashboards to go
`localhost:3000`

That's it! You should now be able to use CiscoDeepVision. If you encounter any issues or have any questions, feel free to open an issue in the repository or reach out to the CiscoDeepVision community!

## ARCHITECTURE:

### Video Source Service:

The Video Source Service is responsible for collecting video streams from various sources, such as cameras or files, and pushing the data into a Redis database. In addition, it stores metadata into another Redis database.

This service is implemented as a Docker container.

### Video Tracking Service:

The Video Tracking Service is based on the open mmlab tracker. It takes an input stream from Redis and adds tracking information to the stream, which is then saved back to Redis.

After the Video Tracking Service has processed the data, consumers are able to view the data in the Redis stream (which now includes tracking information) and the metadata through Grafana dashboards and annotated video servers, as well as any Android or React applications.

Similar to the Video Source Service, the Video Tracking Service and Redis databases are also running as Docker containers.

## PRODUCER:

This is a Python script that allows the user to capture video frames from either a webcam or a local video file and send them to a Redis server to create a video stream. Here is a summary of the code's architecture:

1. The code imports argparse, cv2, logging, redis, time, and pickle libraries. It also imports the urlparse module from the urllib.parse library.
2. The code defines a Video class, which initializes the video capture object, sets video resolution and frame rate, and implements the iteration method that reads the video frames.
3. The code parses command-line arguments using the argparse library.
4. The code establishes a Redis connection using the Redis library and sets up the video source depending on the arguments passed in.
5. The code captures video frames, encodes them as a binary string using the pickle library, and sends them to the Redis server using the xadd method

## TRACKER:

1. The code uses the mmtracking library to perform multi-object tracking on video frames. The mmtracking library provides a high-level API for multi-object tracking, making it easier to implement complex tracking algorithms.
2. The code accepts command-line arguments using the argparse library. The command-line arguments specify the input stream key for the video frames, the class category of the objects to track, the output stream key for the tracklets, the checkpoint file for the model, the device used for inference, the Redis URL for storing the data, and the maximum length of the output stream.
3. The code uses Redis as a data store to manage the input and output streams of video frames and tracklets. Redis is a popular in-memory data store that is often used in real-time data processing applications.
4. The code uses a custom GPUCalculator class and MMTMonitor class to monitor the GPU usage and track the latency of model inference and bounding box calculations. These classes provide valuable insights into the performance of the tracking algorithm and help identify potential bottlenecks in the system.
5. The code uses the results2outs function to convert the output of the tracking algorithm into a dictionary format that can be easily serialized and stored in Redis. The function extracts the bounding box coordinates and object IDs from the tracking results and stores them in a list of dictionaries. The list of dictionaries is then converted to a JSON string and stored in Redis as a stream of tracklets.

## SERVER:

1. RedisImageStream class is defined to fetch the latest image and tracking information from Redis database.
2. A Flask web application is used to create a video stream of the image feed by continuously sending the latest image to the client.
3. TrackletManager class is used to manage the tracking information of the objects in the image feed.
4. The gen() function is used to generate a continuous stream of JPEG frames, which are then sent to the client using the Response class.
5. The code uses various libraries such as OpenCV, PIL, NumPy, and seaborn for image processing, color palette generation, and drawing bounding boxes and tails on the objects being tracked in the image feed.

## CI/CD:

Creating a CI/CD pipeline is an effective way to automate and streamline the software development process. It consists of multiple stages such as code integration, automated testing, code deployment, and more. This makes it easier for developers to track progress, reduce release time, and ensure that code changes are deployed quickly and efficiently. CI/CD also enables teams to continuously improve the quality of the software and makes it easier to identify and fix any potential issues. By harnessing the power of CI/CD, teams can achieve faster, more reliable software development cycles.

### CI Setup:

1. Create GitHub Action CI workflow.
2. Configure. Yaml workflow file as per specification requirements.
3. 	For Instance, set branch rule so when code changes are pushed to the repository, GitHub Action will trigger the configured CI/CD Pipeline/workflow to achieve CI/CD.

### GitHub Action CI workflow inside steps:

Set up Python 3.8

	- Installing python 3.8 version.
Install dependencies.

	- Using requirements.txt to install the libraries used in code repo.
Lint with flake8

	- Default
Test with Unit test via Pytest

	- Running all tests under unittests folder on repo.
 
GitHub Actions tab will show the CI/CD pipeline runs.
Once configured workflow Job run successfully it will build the application.
After build steps complete deploy, step will run to deploy the application to the desire server.

###	Deploy to server follow below steps:
-	Login to the desired server.
-	Create deploy.sh file and copy the contents from deploy.sh placed in CiscoDeepVision root folder.
-	Run deploy.sh script with -b <branch_name> as parameter.
-	Deployment will be success at the same level of deploy.sh script.

### Error Troubleshoot:
-	Github action will shows running jobs and error message if any.
-	Resolve the error which cause build job to be failed 
-	After fix the error job will rerun and make sure application it out of errors/bug.

Note: Entire CI/CD workflow Images attached below which includes CD step which is not automated yet for Ciscodeepvision.



