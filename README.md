# DeepVision

Deep Vision is a serverless edge platform for explainable perception challenges that is focused on enabling the development and deployment of new computer vision and multi-modal spatio-temporal algorithms.  

Deep Vision targets several key problems including improving analytics accuracy on spatio-temporal data, self-supervised learning, lifelong learning and creating explainable models.  

We currently have 3 Deep Vision applications, RecalM, VQPy interface, and Ethosight (integration in process). 

### Getting Started

The system can be run in any environment supporting GPU and docker (NVIDIA [extension](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html) as well). Make sure you have updated GPU driver on the host machine.  
The detailed instructions can be found [here](docs/getstarted.md).  

![getstarts](docs/images/getstarts.gif)

Directory structure:
```
DeepVision
 ├── CONTRIBUTING.md (doc describing contibution on this repo)
 ├── LICENSE (A license used)
 ├── README.md (enty point for docs)
 ├── .gitignore
 ├── .gitmodules (configuration for git submodules)
 ├── .env (tracking model and corresponding accuracy score configuration)
 ├── docker-compose.yml (docker compose configuration)
 ├── Dockerfile (docker file for video sourcing and rendering)
 ├── mkdocs.yml (ReadTheDocs extension configuration)
 ├── producer.py (video reading and sourcing script)
 ├── requirements.txt (dependency requirements)
 ├── server.py (annotated video rendering server)
 ├── trackertotimeseries.py (timeseries labeling)
 ├── .github (github CI/CD workflow configuration)
 ├── dashboards (directory containing metrics dashboard configs for manual setting in grafana)
 ├── data (directory containing video source examples)
 ├── docs (documentation)
 ├── grafana (grafana provisioning and configuration)
 ├── recallm (system for temporal context understanding with NLP)
 ├── videoquery (express query on video (supported with VQPy))
 ├── tracking (containing tacking module and submudules (MMtracking))
 └── tracklet (tracking utility data structures and classes)
```

### RecallM
[RecallM](./recallm/) provides a system capable of natural language analytics by supplementing Large Language Models with an updatable, persistent memory mechanism. Click [here](./recallm/) for more details.

### Documentation

Detailed documentation can be found [here](docs/index.md).

### Support

Any feedback, questions, and issue reports are welcomed. Please follow [Contributor Guide](CONTRIBUTING.md) for more information.


For more details about this and other Cisco Research projects, please visit our home page at [DeepVision Home Page](https://research.cisco.com/research-projects/deep-vision)
