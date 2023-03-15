# CiscoDeepVision

Getting started: 
To clone the repo. Use command:
git clone --recurse-submodules (repo link)

Forward ports: 3000 (grafana), 5001 (server.py), 6380, 6379

Run Producer:
cd CiscoDeepVision
Command: sudo bash start.sh

Run Tracker:
cd tracking
Command: sudo bash -x run.sh -i camera:0 -o camera:0:mot -c PERSON -r redis://redis:6379

Run server.py:
Once you run producer, server.py should run automatically.
Use 5001 port to access it.
Incase it is not running, use Command: python3 server.py
when you access the link from port 5001, add /video as a path to the end of URL to see the results.

For Grafana Dashboard:
Step1:
Run these commands to spin up grafana and redis time series container.
Command: sudo docker run -d --name=grafana -p 3000:3000 --net redisconnection grafana/grafana 
Command: sudo docker run --name redis -p 6380:6379 --network redisconnection redislabs/redistimeseries
Step 2:
See wiki/grafana_readme.docx for detailed instruction to set up grafana.
