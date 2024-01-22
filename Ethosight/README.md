# Ethosight

<!-- TODO: Add a brief introduction about Ethosight here -->

## Installation

### Setting Up the Environment

> **Note:** The conda environment is currently named `imagebind`. This will be changed to `ethosight` in a future update.

1. If you're setting up for the first time, create the environment using the provided `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    ```

2. If you're updating an existing environment, use the following command instead:

    ```bash
    conda env update --name imagebind --file environment.yml
    ```

### Installing Ethosight

After setting up the environment, you can install Ethosight using pip:

```bash
pip install -e .
```

## Running the System

Before running the system, ensure that the Consul server is running. You can start the Consul server with the following command:
```bash
consul agent -dev
```

Once the Consul server is running, you can view the Consul UI in a web browser by navigating to:

[http://localhost:8500](http://localhost:8500)

Next, ensure that the 'ethosight-nginx' Docker container is running. This container runs the Nginx server. You can use the `run_nginx.sh` script to continuously run an Nginx server in a Docker container named 'ethosight-nginx'. 

Here's how to run the `run_nginx.sh` script:

```bash
./run_nginx.sh
```

After the 'ethosight-nginx' Docker container is running, you can execute the `run_consultemplate.sh` script. This script uses consul-template to dynamically update the Nginx configuration file based on the `nginx.template`. After the configuration file is updated, the script triggers a reload of the Nginx configuration inside the 'ethosight-nginx' Docker container.

You can run this script with the following command:

```bash
./run_consultemplate.sh
```

## Running Ethosight GPU Servers

To run Ethosight GPU Servers, you can use the `runserver.sh` script with the `runserver` argument. This script starts a server that can process requests from the Ethosight system.

Here's how to run the `runserver.sh` script:

```bash
./runserver.sh runserver
```

## Utilizing Multiple GPU Servers

The Ethosight system is designed to scale and can utilize any number of GPU servers on multiple hosts. This is achieved using Consul for service discovery.

Each GPU server should run the `runserver.sh` script as described in the "Running Ethosight GPU Servers" section. When a GPU server starts, it registers itself with the Consul server. The Nginx server uses the information from the Consul server to distribute requests among the available GPU servers.

To add a new GPU server to the system, simply start the `runserver.sh` script on the host where the GPU server is located. The new GPU server will automatically register itself with the Consul server and start processing requests from the Ethosight system.

## Running the Django Server

Before starting the Django server, ensure that the PostgreSQL server is running. You can use the `runpostgress.sh` script to start a PostgreSQL server in a Docker container named 'djangopostgres'.

Here's how to run the `runpostgress.sh` script:

```bash
./runpostgress.sh
``````

Once the PostgreSQL server is running, you can start the Django server. Make sure you're in the directory where `manage.py` is located, then run the following command:

```bash
python manage.py runserver 8080
```

This command starts the Django development server. By default, the server runs on `localhost` on port `8000`. You can view your application by navigating to `http://localhost:8080` in your web browser.

