#!/usr/bin/env bash

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

echo "__version__ = '${RELEASE_TAG}'" > $TRAINER_PACKAGE_PATH/__init__.py

now=$(date +"%Y%m%d_%H%M%S") 
JOB_NAME="rpivision_${now}"

gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region $REGION \
    --python-version 3.5 \
    --runtime-version 1.10