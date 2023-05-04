# CiscoDeepVision

## INTRODUCTION:

CiscoDeepVision is a comprehensive video analytics system that enables users to monitor and analyze video streams from various sources with ease



## ARCHITECTURE:  

![architecture](docs/images/architecture.png)  

detailed info is [here](docs/architecture.md)

## CI/CD:

Creating a CI/CD pipeline is an effective way to automate and streamline the software development process. It consists of multiple stages such as code integration, automated testing, code deployment, and more. This makes it easier for developers to track progress, reduce release time, and ensure that code changes are deployed quickly and efficiently. CI/CD also enables teams to continuously improve the quality of the software and makes it easier to identify and fix any potential issues. By harnessing the power of CI/CD, teams can achieve faster, more reliable software development cycles.

### CI Setup:

1. Create GitHub Action CI workflow.
2. Configure. Yaml workflow file as per specification requirements.
3. 	For Instance, set branch rule so when code changes are pushed to the repository, GitHub Action will trigger the configured CI/CD Pipeline/workflow to achieve CI/CD.

### GitHub Action CI workflow inside steps:

Set up Python 3.8

	- Installing python 3.8 version.
Install dependencies.

	- Using requirements.txt to install the libraries used in code repo.
Lint with flake8

	- Default
Test with Unit test via Pytest

	- Running all tests under unittests folder on repo.
 
GitHub Actions tab will show the CI/CD pipeline runs.
Once configured workflow Job run successfully it will build the application.
After build steps complete deploy, step will run to deploy the application to the desire server.

###	Deploy to server follow below steps:
-	Login to the desired server.
-	Create deploy.sh file and copy the contents from deploy.sh placed in CiscoDeepVision root folder.
-	Run deploy.sh script with -b <branch_name> as parameter.
-	Deployment will be success at the same level of deploy.sh script.

### Error Troubleshoot:
-	Github action will shows running jobs and error message if any.
-	Resolve the error which cause build job to be failed 
-	After fix the error job will rerun and make sure application it out of errors/bug.

Note: Entire CI/CD workflow Images attached below which includes CD step which is not automated yet for Ciscodeepvision.



