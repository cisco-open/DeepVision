FROM python:3.7

WORKDIR /app
RUN apt-get update && apt-get install -y python3-opencv

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
