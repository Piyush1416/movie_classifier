# using the ubuntu:latest image from docker-hub
FROM ubuntu:latest

# basic update all command
RUN apt-get update -y

# installing the python, pip and some necessary python package dependencies
RUN apt-get install -y python-pip python-dev build-essential

# add the whole current project content to car-project directory on docker image
ADD . /movie_classifier1.0

# make it as current working directory to run further commands
WORKDIR /movie_classifier1.0

# copy the requirements.txt to docker so that everytime we run this requirements will be already available
COPY requirements.txt .

# install the requirements
RUN pip install -r requirements.txt

CMD /bin/bash install.sh
