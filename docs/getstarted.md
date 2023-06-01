## GETTING STARTED:

### Cloning the repo

While cloning the repo use the command:
```bash
git clone --recurse-submodules https://github.com/cisco-open/DeepVision.git
```

### Running  and building with ./run script
This script run is a bash script designed to manage Docker services for the DeepVision project. It allows you to start (up), stop (down), or build (--build) different Docker service profiles.

The script uses the DOCKER_BUILDKIT environment variable to enhance the Docker build process. The profiles for the services are defined in an associative array in the script. Each service profile has a corresponding description.
Usage

./run <profilename> [up|down] [--build]

### Parameters:

    <profilename>: The profile of the service you want to manage. This should match one of the keys in the associative array profiles.
    [up|down]: Optional argument to start or stop the Docker services in the profile. If not provided, the default operation is 'up'.
    [--build]: Optional argument to build the Docker images for the services in the profile.

### Service Profiles:

The script includes the following service profiles:

    tracking: The tracking service including redis, redistimeseries, grafana, tracker, and producer.
    haidetection: The hybridaiobjectdetection service.
    ona: The ONA service.
    haitracking: (placeholder) The hybridaiobjecttracking service.
    haitext: (placeholder) The hybridaistreamingtext service.
    hainetworking: (placeholder) The hybridainetworking service.

### Examples:

    To start the tracking service profile:

./run tracking up

    To stop the haidetection service profile:

./run haidetection down

    To build and start the ona service profile:

./run ona up --build

Note: If an invalid profile name is given, the script will display a "Invalid profile" error message and print the help message.

### Configuration
It's pretty simple to configure and change tracking model and corresponding accuracy score using parameters inside `.env` file in the root directory. Just use commenting and uncommenting approach.  
There are other configurable properties described in `.env` file (details are placed in the file itself).

### For Webserver and grafana dashboard

To see annotated video server and grafana dashboards go
`localhost:3000`

That's it! You should now be able to use DeepVision.  

![getstarts](images/getstarts.gif)