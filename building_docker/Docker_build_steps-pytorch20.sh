#!/bin/bash

## STEP 1
docker buildx build \
  -f Dockerfile-pytorch20 \
  --platform linux/amd64 \
  --target step1 \
  -t jacobdavidson/beesbook-step1:cuda11-py3.11 \
  .

## STEP 2
docker buildx build \
  -f Dockerfile-pytorch20 \
  --platform linux/amd64 \
  --target step2 \
  -t jacobdavidson/beesbook-step2:cuda11-py3.11 \
  .

## STEP 3
docker buildx build \
  -f Dockerfile-pytorch20 \
  --platform linux/amd64 \
  --target step3 \
  -t jacobdavidson/beesbook-step3:cuda11-py3.11 \
  .

## STEP 4
docker buildx build \
  -f Dockerfile-pytorch20 \
  --platform linux/amd64 \
  --target step4 \
  -t jacobdavidson/beesbook-step4:cuda11-py3.11 \
  .

## STEP 5
docker buildx build \
  -f Dockerfile-pytorch20 \
  --platform linux/amd64 \
  --target step5 \
  -t jacobdavidson/beesbook-step5:cuda11-py3.11 \
  .

## STEP 6
docker buildx build \
  -f Dockerfile-pytorch20 \
  --platform linux/amd64 \
  --target step6 \
  -t jacobdavidson/beesbook-step6:cuda11-py3.11 \
  .

## STEP 7
docker buildx build \
  -f Dockerfile-pytorch20 \
  --platform linux/amd64 \
  -t jacobdavidson/beesbook:cuda11-py3.11 \
  .

## Finishing and pushing
docker tag jacobdavidson/beesbook:cuda11-py3.11 jacobdavidson/beesbook:cuda11-py3.11
# docker login
docker push jacobdavidson/beesbook:cuda11-py3.11