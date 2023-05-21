# DeepVision

Deep Vision is a serverless edge platform for explainable perception challenges that is focused on enabling the development and deployment of new computer vision and multi-modal spatio-temporal algorithms.  

Deep Vision targets several key problems including improving analytics accuracy on spatio-temporal data, self-supervised learning, lifelong learning and creating explainable models.  

### Getting Started

The system can be run in any environment supporting GPU and docker (NVIDIA [extension](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html) as well).  
The detailed instructions can be found [here](docs/getstarted.md).  

Directory structure:
```
DeepVision
 ├── CONTRIBUTING.md (doc describing contibution on this repo)
 ├── LICENSE (A license used)
 ├── README.md (enty point for docs)
 ├── .gitignore
 ├── .gitmodules (config for git submodules)
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
 ├── tracking (containing tacking module and submudules (MMtracking))
 └── tracklet (tracking utility data structures and classes)
```

### Documentation

Detailed documentation can be found [here](docs/index.md).

### Support

Any feedback, questions, and issue reports are welcomed. Please follow [Contributor Guide](CONTRIBUTING.md) for more information.


