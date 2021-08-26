#!/bin/bash

cd ..
docker build -t parsebag .
cd docker-tests
docker build -t parsebag:test .
docker run --rm parsebag:test
