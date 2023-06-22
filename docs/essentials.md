## PRODUCER:

This is a Python script that allows the user to capture video frames from either a webcam or a local video file and send them to a Redis server to create a video stream. Here is a summary of the code's architecture:

1. The code imports argparse, cv2, logging, redis, time, and pickle libraries. It also imports the urlparse module from the urllib.parse library.
2. The code defines a Video class, which initializes the video capture object, sets video resolution and frame rate, and implements the iteration method that reads the video frames.
3. The code parses command-line arguments using the argparse library.
4. The code establishes a Redis connection using the Redis library and sets up the video source depending on the arguments passed in.
5. The code captures video frames, encodes them as a binary string using the pickle library, and sends them to the Redis server using the xadd method


## SERVER:

1. RedisImageStream class is defined to fetch the latest image and tracking information from Redis database.
2. A Flask web application is used to create a video stream of the image feed by continuously sending the latest image to the client.
3. TrackletManager class is used to manage the tracking information of the objects in the image feed.
4. The gen() function is used to generate a continuous stream of JPEG frames, which are then sent to the client using the Response class.
5. The code uses various libraries such as OpenCV, PIL, NumPy, and seaborn for image processing, color palette generation, and drawing bounding boxes and tails on the objects being tracked in the image feed.

## TRACKER

The main logic is placed inside `tracking/tracker.py` script. This Python script is a program for tracking objects in videos using the MMTracking library.  
1. The script defines a class called Tracker, which represents the object tracker. The Tracker class has the following attributes:  
    * `model`: The tracking model initialized using the init_model function.  
    * `xreader_writer`: An instance of the RedisStreamXreaderWriter class for reading from and writing to Redis streams.  
    * `model_run_monitor`: An instance of the MMTMonitor class for monitoring the model's runtime latency.  
    * `bbox_run_monitor`: An instance of the MMTMonitor class for monitoring the bounding boxes' runtime latency.  
    * `gpu_calculator`: An instance of the GPUCalculator class for calculating GPU usage.  
    * `ts_manager`: An instance of the TSManager class for managing Redis TimeSeries.
    * `class_id`: The class category of the tracked objects.
2. The Tracker class also has several methods:  
   * `_get_frame_data`: A private method that extracts the frame ID and image data from a Redis stream message.  
   * `_construct_response`: A private method that constructs a response dictionary containing the tracking information for a frame.  
   * `results2outs`: A method that converts the tracking results into a dictionary format.
   *  `inference`: The main method responsible for performing the object tracking inference. It continuously reads messages from the Redis stream, performs tracking inference on the received frames, and writes the tracking results back to the stream.  
3. The script defines a custom JSON encoder class called `NpEncoder`, which extends json.JSONEncoder. This class is used to handle encoding of NumPy data types in JSON serialization.  
4. The `main` function is defined, which serves as the entry point of the script. It sets up the argument parser, parses the command-line arguments, and initializes the necessary components such as Redis connections, monitor instances, and the Tracker object. Finally, it calls the inference method of the Tracker object to start the tracking process.  



## ACTION RECOGNIZER:

The main logic is placed inside `actionrecognition/actionrec.py` script. This script sets up an action recognition pipeline using the MMAction2 framework and performs real-time action recognition on frames obtained from a Redis stream.  

1. The script defines a class called ActionRecognizer with the following methods and attributes:
   * __init__(self, action_inferencer, xreader_writer, algo, sample_size, batch_size, top_pred_count): Initializes the ActionRecognizer object with an action inferencer, a Redis stream reader and writer, the name of the algorithm, sample size, batch size, and top prediction count.
   * ___get_frame_data__(self, data): Extracts the image data from the input data dictionary and returns it.
   * ___output_result__(self, prediction, labels): Processes the prediction scores, sorts them, selects the top prediction labels, and returns them as a list of tuples containing the label and its corresponding score.
   * ___init_labels__(self, rec): Initializes the labels attribute by reading label data from a file based on the dataset used by the algorithm.
   * __run__(self): Runs the action recognition process in a loop, continuously reading frames from the Redis stream, performing inference on a sample of frames, and writing the action recognition results back to the Redis stream.

2. The script defines a `main` function:  
   * It creates an argument parser and defines command-line arguments for the algorithm name, input and output stream keys, device, batch size, sample size, Redis URL, and maximum length of the output stream.  
   * It parses the command-line arguments.  
   * It parses the Redis URL and establishes a connection to Redis.
   * It creates a RedisStreamXreaderWriter object with the input and output stream keys and the Redis connection.  
   * It creates an MMAction2Inferencer object with the algorithm name, device, and input format.  
   * It creates an ActionRecognizer object with the action inferencer, Redis stream reader and writer, algorithm name, sample size, batch size, and top prediction count.  
   * It calls the `run` method of the ActionRecognizer object.
   * The script executes the `main` function if it is being run as the main module.  
