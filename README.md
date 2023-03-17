##Cloning the repo
While cloning the repo use the command:
git clone --recurse-submodules "link"

## Running  and building

Launch 1st command for build and run  
Launch 2nd command for run  

1.`docker-compose up --build`  
2.`docker-compose up`  

##For grafana dashboard
Import the dashboards from dashboards folder. Json file is present. 
To import the json file, please see wiki/grafana/grafana_dashboard for further information
Make sure ports: 5001, 3000, 6379 and 6380 are forwarded.

Open the browser and go to `localhost:5002/video` and see the result  
If it doesnot work, try `localhost:5001/video` 


