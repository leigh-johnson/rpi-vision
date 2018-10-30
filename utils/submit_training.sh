#!/usr/bin/env bash

now=$(date +"%Y%m%d_%H%M%S") 
JOB_NAME="rpivision_$now"
JOB_ID="rpivision_$now"
gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region $REGION \
    --python-version 3.5 \
    --runtime-version 1.10