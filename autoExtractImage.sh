#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name extractImage.json?[yes/no]:"
read which
while [ ! $which = "yes" -a ! $which = "no" ]
do
 echo -n "Is json file name the same as this file name?[yes/no]:"
 read which
done

# Specify json file.
if [ $which = "yes" ];then
 JSON_NAME="extractImage.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

# From json file, read required variables.
readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"
readonly DATA_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".data_directory"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly IMAGE_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".image_patch_size")
readonly LABEL_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".label_patch_size")
readonly SLIDE=$(cat ${JSON_FILE} | jq -r ".slide")
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")
readonly LOG_FILE=$(eval echo $(cat ${JSON_FILE} | jq -r ".log_file"))
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly LABEL_NAME=$(cat ${JSON_FILE} | jq -r ".label_name")
readonly MASK_NAME=$(cat ${JSON_FILE} | jq -r ".mask_name")

echo "DATA_DIRECTORY:${DATA_DIRECTORY}"
echo "SAVE_DIRECTORY:${SAVE_DIRECTORY}"
echo "LOG_FILE:${LOG_FILE}"

# Make directory to save LOG.
mkdir -p `dirname ${LOG_FILE}`
date >> $LOG_FILE

for number in ${NUM_ARRAY[@]}
do
 data="${DATA_DIRECTORY}/case_00${number}"
 image="${data}/${IMAGE_NAME}"
 label="${data}/${LABEL_NAME}"
 mask="${data}/${MASK_NAME}"
 save="${SAVE_DIRECTORY}/case_00${number}"

 echo "Image:${image}"
 echo "Label:${label}"
 echo "Mask:${mask}"
 echo "Save:${save}"
 echo "IMAGE_PATCH_SIZE:${IMAGE_PATCH_SIZE}"
 echo "LABEL_PATCH_SIZE:${LABEL_PATCH_SIZE}"

 python3 extractImage.py ${image} ${label} ${save} --mask_path ${mask} --image_patch_size ${IMAGE_PATCH_SIZE} --label_patch_size ${LABEL_PATCH_SIZE} --slide ${SLIDE}

 # Judge if it works.
 if [ $? -eq 0 ]; then
  echo "case_00${number} done."
 
 else
  echo "case_00${number}" >> $LOG_FILE
  echo "case_00${number} failed"
 
 fi

done

