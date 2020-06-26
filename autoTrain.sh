#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name train.json?[yes/no]:"
read which
while [ ! $which = "yes" -a ! $which = "no" ]
do
 echo -n "Is json file name the same as this file name?[yes/no]:"
 read which
done

# Specify json file path.
if [ $which = "yes" ];then
 JSON_NAME="train.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

readonly DATASET_PATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".dataset_path"))
readonly MODEL_SAVEPATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".model_savepath"))
readonly TRAIN_LIST=$(cat ${JSON_FILE} | jq -r ".train_list")
readonly VAL_LIST=$(cat ${JSON_FILE} | jq -r ".val_list")
readonly MODULE_NAME=$(cat ${JSON_FILE} | jq -r ".module_name")
readonly SYSTEM_NAME=$(cat ${JSON_FILE} | jq -r ".system_name")
readonly CHECKPOINT_NAME=$(cat ${JSON_FILE} | jq -r ".checkpoint_name")
readonly LOG=$(eval echo $(cat ${JSON_FILE} | jq -r ".log"))
readonly IN_CHANNEL=$(cat ${JSON_FILE} | jq -r ".in_channel")
readonly NUM_CLASS=$(cat ${JSON_FILE} | jq -r ".num_class")
readonly LEARNING_RATE=$(cat ${JSON_FILE} | jq -r ".learning_rate")
readonly BATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".batch_size")
readonly NUM_WORKERS=$(cat ${JSON_FILE} | jq -r ".num_workers")
readonly EPOCH=$(cat ${JSON_FILE} | jq -r ".epoch")
readonly GPU_IDS=$(cat ${JSON_FILE} | jq -r ".gpu_ids")
readonly API_KEY=$(cat ${JSON_FILE} | jq -r ".api_key")
readonly PROJECT_NAME=$(cat ${JSON_FILE} | jq -r ".project_name")
readonly EXPERIMENT_NAME=$(cat ${JSON_FILE} | jq -r ".experiment_name")

echo "DATASET_PATH:${DATASET_PATH}"
echo "MODEL_SAVEPATH:${MODEL_SAVEPATH}"
echo "TRAIN_LIST:${TRAIN_LIST}"
echo "VAL_LIST:${VAL_LIST}"
echo "MODULE_NAME:${MODULE_NAME}"
echo "SYSTEM_NAME:${SYSTEM_NAME}"
echo "CHECKPOINT_NAME:${CHECKPOINT_NAME}"
echo "LOG:${LOG}"
echo "IN_CHANNEL:${IN_CHANNEL}"
echo "NUM_CLASS:${NUM_CLASS}"
echo "LEARNING_RATE:${LEARNING_RATE}"
echo "BATCH_SIZE:${BATCH_SIZE}"
echo "NUM_WORKERS:${NUM_WORKERS}"
echo "EPOCH:${EPOCH}"
echo "GPU_IDS:${GPU_IDS}"
echo "API_KEY:${API_KEY}"
echo "PROJECT_NAME:${PROJECT_NAME}"
echo "EXPERIMENT_NAME:${EXPERIMENT_NAME}"

python3 train.py ${DATASET_PATH} ${MODEL_SAVEPATH} ${MODULE_NAME} ${SYSTEM_NAME} ${CHECKPOINT_NAME} --train_list ${TRAIN_LIST} --val_list ${VAL_LIST} --log ${LOG} --in_channel ${IN_CHANNEL} --num_class ${NUM_CLASS} --lr ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --epoch ${EPOCH} --gpu_ids ${GPU_IDS} --api_key ${API_KEY} --project_name ${PROJECT_NAME} --experiment_name ${EXPERIMENT_NAME} 
