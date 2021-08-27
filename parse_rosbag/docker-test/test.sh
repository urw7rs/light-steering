#!/bin/bash

cd ..
docker build -t parsebag .
cd docker-test
docker build -t parsebag:test .
docker run --rm parsebag:test
