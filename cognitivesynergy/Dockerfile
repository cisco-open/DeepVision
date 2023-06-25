FROM ubuntu:22.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list
RUN apt-get update

# Install Python 3 and pip
RUN apt-get install -y python3 python3-pip

# Install build-essential, etc...
RUN apt install -y build-essential vim python3 python3-pip \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

# Download Docker's official GPG key and verify the integrity of the package
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

# Add Docker's official repository
RUN add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

# Install Docker CLI only
RUN apt-get update && apt-get install -y docker-ce-cli

RUN apt-get install -y fonts-freefont-ttf

# Verify Docker client installation
RUN docker --version

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
CMD ["python", "main.py"]
