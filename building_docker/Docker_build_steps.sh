#!/bin/bash
set -e

# New value each run, so step4 re-installs the bb_* repos from GitHub and every later
# stage is rebuilt on top of it (see ARG BB_CACHEBUST in the Dockerfile).
BB_CACHEBUST=$(date +%Y%m%d-%H%M%S)

## STEP 1
docker buildx build \
  --platform linux/amd64 \
  --target step1 \
  -t jacobdavidson/beesbook-step1:latest \
  .

## STEP 2
docker buildx build \
  --platform linux/amd64 \
  --target step2 \
  -t jacobdavidson/beesbook-step2:latest \
  .

## STEP 3
docker buildx build \
  --platform linux/amd64 \
  --target step3 \
  -t jacobdavidson/beesbook-step3:latest \
  .

## STEP 4
docker buildx build \
  --platform linux/amd64 \
  --target step4 \
  --build-arg BB_CACHEBUST=$BB_CACHEBUST \
  -t jacobdavidson/beesbook-step4:latest \
  .

## STEP 5
docker buildx build \
  --platform linux/amd64 \
  --target step5 \
  --build-arg BB_CACHEBUST=$BB_CACHEBUST \
  -t jacobdavidson/beesbook-step5:latest \
  .

## STEP 6
docker buildx build \
  --platform linux/amd64 \
  --target step6 \
  --build-arg BB_CACHEBUST=$BB_CACHEBUST \
  -t jacobdavidson/beesbook-step6:latest \
  .

## FINAL IMAGE
docker buildx build \
  --platform linux/amd64 \
  --build-arg BB_CACHEBUST=$BB_CACHEBUST \
  -t jacobdavidson/beesbook:latest \
  .


## Finishing and pushing
# docker login
docker push jacobdavidson/beesbook:latest

## Also push to K8s registry
docker tag jacobdavidson/beesbook:latest ccu-k8s.inf.uni-konstanz.de:32250/jacob.davidson/beesbook:latest
docker push ccu-k8s.inf.uni-konstanz.de:32250/jacob.davidson/beesbook:latest
## Notes
# add --load in order to be able to access the build step in successive steps