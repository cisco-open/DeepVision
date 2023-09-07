# Video Query

The video query service enables users to express their objects/events of interest on video and obtain query results. It is built upon the VQPy framework, which is an object-oriented video query framework based on Python. For more information about the VQPy framework, please refer to the [VQPy GitHub repository](https://github.com/vqpy/vqpy). 

Different video queries can be easily expressed through the constructs provided by VQPy. An example application that demonstrates the detection of loitering behavior and raises an alarm is provided in the "loitering.py" file.


## Running the Project

First, make sure you are under the root repository of the DeepVision project.

To start the service:  ```./run query up videoquery/.env```

To build and start the service: ```./run query up videoquery/.env --build```

To shutdown the service: ```./run query down```

## Visualization

By clicking on the "Video Query" dashboard, you can visualize the query results.

![loitering-Visualization](./loitering-demo-19s.gif)