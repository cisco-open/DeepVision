# Ethosight

<!-- TODO: Add a brief introduction about Ethosight here -->

## Installation

### Setting Up the Environment

> **Note:** The conda environment is currently named `imagebind`. This will be changed to `ethosight` in a future update.

1. If you're setting up for the first time, create the environment using the provided `environment.yml` file in `/install` subfolder:

    ```bash
    conda env create -f install/environment.yml
    ```

2. If you're updating an existing environment, use the following command instead:

    ```bash
    conda env update --name imagebind --file install/environment.yml
    ```
3. Then activate the conda environmet:  
   ```
   conda activate imagebind
   ```
### Installing Ethosight

After setting up the environment, you can install Ethosight using pip:

```bash
pip install -e .
```

## Running the System

Before running the system, ensure that you have installed [Consul](https://developer.hashicorp.com/consul/install) and [consul-template](https://github.com/hashicorp/consul-template?tab=readme-ov-file#installation), and ensure that the Consul server is running. You can start the Consul server with the following command:
```bash
consul agent -dev
```

Once the Consul server is running, you can view the Consul UI in a web browser by navigating to:

[http://localhost:8500](http://localhost:8500)

Next, ensure that the 'ethosight-nginx' Docker container is running. This container runs the Nginx server. You can use the `run_nginx.sh` script to continuously run an Nginx server in a Docker container named 'ethosight-nginx'. 

Here's how to run the `run_nginx.sh` script:

```bash
./bin/run_nginx.sh
```

After the 'ethosight-nginx' Docker container is running, you can execute the `run_consultemplate.sh` script. This script uses consul-template to dynamically update the Nginx configuration file based on the `nginx.template`. After the configuration file is updated, the script triggers a reload of the Nginx configuration inside the 'ethosight-nginx' Docker container.

You can run this script with the following command:

```bash
./bin/run_consultemplate.sh
```

## Running Ethosight GPU Servers

Before running you can modify and set up environment variables in `bin/env_variables_setup.sh` to be appropriate for your system.
Then source them to your session.

`source ./bin/env_variables_setup.sh`

To run Ethosight GPU Servers, you can use the `runserver.sh` script with the `runserver` argument. This script starts a server that can process requests from the Ethosight system.

Here's how to run the `runserver.sh` script:

```bash
./bin/runserver.sh runserver
```

## Utilizing Multiple GPU Servers

The Ethosight system is designed to scale and can utilize any number of GPU servers on multiple hosts. This is achieved using Consul for service discovery.

Each GPU server should run the `runserver.sh` script as described in the "Running Ethosight GPU Servers" section. When a GPU server starts, it registers itself with the Consul server. The Nginx server uses the information from the Consul server to distribute requests among the available GPU servers.

To add a new GPU server to the system, simply start the `runserver.sh` script on the host where the GPU server is located. The new GPU server will automatically register itself with the Consul server and start processing requests from the Ethosight system.

## Running the Django Server

Before starting the Django server, ensure that the PostgreSQL server is running. You can use the `runpostgress.sh` script to start a PostgreSQL server in a Docker container named 'djangopostgres'.

Here's how to run the `runpostgress.sh` script:

```bash
./website/runpostgress.sh
``````

Once the PostgreSQL server is running, you can start the Django web app inside /website folder. Run the following command:

```bash
./website/runwebapp.sh
```

If no need to create a super user and migrations are done, you can simply run the application  

```bash
python website/manage.py runserver 8080
```
You can view your application by navigating to `http://localhost:8080` in your web browser.  
You can create your own ethosight [configuration](#ethosight-configuration) by accessing `http://localhost:8080/admin`.
You can find an example in `./configs` folder.  

## Registering new users.
For registering new users, you have to provide mail sending environment variables in `bin/env_variables_setup.sh` 
and request an access code in the registration form. The admin users should approve pending users from the admin panel.  
If there's no capability to handle mail sending, you can generate
access codes manually using `genaccesscodes.py`, and use that access code in the form without requesting.

## Ethosight Configuration

`ethosight.yml` file is the setup configuration for the ethosight application. 
You can find the example file inside `./configs` folder with all possible configurations and their explanations.


## CLI
Besides main application represented as UI. Ethosight provides CLI for all core classes and functionalities like
EthosightAppCLI, EthosightCLI, EthosightDatasetCLI, EthosightMediaAnalyzerCLI.

The main one is EthosightAppCLI with bunch of useful methods. Some of them are still on implementation.

* create_app (app_dir, config_file) - creates new application
  * app_dir - the location where the application will be created and run along with config files, embeddings, labels
  * config_file - *.yml config file path 
* delete_app (app_dir) - deletes application located in app_dir
  * app_dir - the application directory
* benchmark (app_dir) - Computes accuracy on a directory of images
  * Computes accuracy on a directory of images.
* optimize(app_dir) - optimizes the EthosightApp
  * app_dir - the application directory
* run(app_dir, image) - Runs the EthosightApp on a single image
  * app_dir - the application directory
  * image - image file path
* benchmark_video (app_dir, video_gt_csv_filename) - Runs video benchmarking on a video
  * app_dir - the application directory
  * video_gt_csv_filename - video file names with ground truths
* rank_affinities (app_dir, json_file_path) - ranks affinities from json results
  * app_dir - the application directory
  * json_file_path - json file path containing already computed affinity scores
* phase2videobenchmarks (app_dir, phase2_groundtruth_csv) - runs benchmarks on all of the affinity score json files contained in the csv file. these are produced by phase1 Ethosight processing of video datasets
  * app_dir - the application directory
  * phase2_groundtruth_csv - the csv file path
* add_labels (app_dir, labels) - Adds new labels to the EthosightApp
  * app_dir - the application directory
  * labels - new labels specified
   