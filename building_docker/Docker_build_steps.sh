#!/bin/bash

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
  -t jacobdavidson/beesbook-step4:latest \
  .

## STEP 5
docker buildx build \
  --platform linux/amd64 \
  --target step5 \
  -t jacobdavidson/beesbook-step5:latest \
  .

## STEP 6
docker buildx build \
  --platform linux/amd64 \
  --target step6 \
  -t jacobdavidson/beesbook-step6:latest \
  .

## FINAL IMAGE
docker buildx build \
  --platform linux/amd64 \
  -t jacobdavidson/beesbook:latest \
  .


## Finishing and pushing
docker tag jacobdavidson/beesbook:latest jacobdavidson/beesbook:latest
# docker login
docker push jacobdavidson/beesbook:latest
## Notes
# add --load in order to be able to access the build step in successive steps