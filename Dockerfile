FROM ubuntu

ADD install.sh
ADD setup.py
ADD /data/*
ADD /movie_classifier/*

RUN apt-get update \
    && install.sh

ENTRYPOINT ["python", "movie_classifier"]