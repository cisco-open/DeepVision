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
See Dashboard/readme.md for detailed instruction to set up grafana.
