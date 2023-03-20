## Cloning the repo

While cloning the repo use the command:
```bash
git clone --recurse-submodules https://github.com/CiscoDeepVision/CiscoDeepVision.git
```

## Running  and building

Launch docker compose

 ```bash
    docker-compose up --build
```

## For grafana dashboard and Webserver

Make sure ports: 5002, 3000, 6379 and 6380 are forwarded.

To see grafana dashboards to go
`localhost:3000`

