# DeepVision Project

DeepVision is an innovative open-source project dedicated to advancing computer vision technology by integrating deep learning, serverless computing, hybrid AI, and intent-based frameworks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need to have Docker and Docker Compose installed on your system. Visit Docker's official website to download Docker Desktop which comes with Docker Compose.

### Setting up the project

1. Clone the repository by running `git clone --recurse-submodules https://github.com/youruser/yourrepo.git`
2. Navigate to the project directory using `cd yourrepo`

### Running the project

To run the project, use the `docker-compose up` command followed by the service name. Here are some examples:

1. To bring up the HybridAIObjectDetection configuration, run `docker-compose up HybridAIObjectDetection`
2. To bring up the HybridAIObjectTracking configuration, run `docker-compose up HybridAIObjectTracking`

If you want to run the services in the background, you can add the `-d` option:

1. To bring up the HybridAIObjectDetection configuration in detached mode, run `docker-compose up -d HybridAIObjectDetection`
2. To bring up the HybridAIObjectTracking configuration in detached mode, run `docker-compose up -d HybridAIObjectTracking`

To stop the services, run `docker-compose down`.

Remember to replace `youruser`, `yourrepo`, `HybridAIObjectDetection`, and `HybridAIObjectTracking` with your actual GitHub username, repository name, and the names of your services, respectively.
