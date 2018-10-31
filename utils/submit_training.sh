#!/usr/bin/env bash

$1

if [ -z "$1" ]
  then
    echo "please specify a training package [shapes, dice]"
fi

# bump version
# get last 3 git tags
echo "Last 3 release tags:"
git tag --sort=-version:refname | head -n 3

# prompt for next tag
echo "Enter next release tag: "
read RELEASE_TAG

# create git tag
if [ $(git tag -l "$RELEASE_TAG") ]; then
    read -r -p "WARNING ${RELEASE_TAG} already exists. Overwrite? [y/N] " OVERWRITE_RES
    case "$OVERWRITE_RES" in
        [yY][eE][sS]|[yY]) 
            git tag -d ${RELEASE_TAG}
            git tag ${RELEASE_TAG}
            ;;
        *)
            exit 1
            ;;
    esac

else
  git tag ${RELEASE_TAG}
fi

sed -i "s/__version__ = '*.*'/__version__ = '${RELEASE_TAG}/g" $TRAINER_PACKAGE_PATH/__init__.py

now=$(date +"%Y%m%d_%H%M%S") 
JOB_NAME="rpivision_${1}_${now}"
TRAINER_PACKAGE_PATH="${HOME}/projects/raspberry-pi-vision/trainers/"
REGION="us-west2"
MAIN_TRAINER_MODULE="trainers.${1}.task"
PACKAGE_STAGING_PATH="gs://raspberry-pi-vision-build"
JOB_DIR="gs://raspberry-pi-vision/job-output"
PACKAGE_NAME="rpivision_${1}"

gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region $REGION \
    --python-version 3.5 \
    --runtime-version 1.10 \
    --
    --REMOTE