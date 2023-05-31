## GETTING STARTED:

### Cloning the repo

While cloning the repo use the command:
```bash
git clone --recurse-submodules https://github.com/CiscoDeepVision/DeepVision.git
```

### Running  and building

Launch docker compose

 ```bash
    docker-compose up --build
```
You may launch `docker compose down` before, to make sure you don't duplicate containers in the same docker network, that can cause some problems.  
If the operating system is Ubuntu, doing `export DOCKER_BUILDKIT=1` may be required.  

### Configuration
It's pretty simple to configure and change tracking model and corresponding accuracy score using parameters inside `.env` file in the root directory. Just use commenting and uncommenting approach.  
There are other configurable properties described in `.env` file (details are placed in the file itself).

### For Webserver and grafana dashboard

To see annotated video server and grafana dashboards go
`localhost:3000`

That's it! You should now be able to use CiscoDeepVision.  

![getstarts](images/getstarts.gif)