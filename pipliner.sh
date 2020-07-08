#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name pipliner.json?[yes/no]:"
read which
while [ ! $which = "yes" -a ! $which = "no" ]
do
 echo -n "Is json file name the same as this file name?[yes/no]:"
 read which
done

# Specify json file path.
if [ $which = "yes" ];then
 JSON_NAME="pipliner.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

# Training input
readonly DATASET_PATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".dataset_path"))
dataset_path="${DATASET_PATH}/image"
save_directory="${DATASET_PATH}/segmentation"

readonly MODEL_SAVEPATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".model_savepath"))

readonly TRAIN_LIST=$(cat ${JSON_FILE} | jq -r ".train_list")
readonly VAL_LIST=$(cat ${JSON_FILE} | jq -r ".val_list")
readonly TEST_LIST=$(cat ${JSON_FILE} | jq -r ".test_list")
test_list=(${TEST_LIST// / })

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

# Segmentation input
readonly DATA_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".data_directory"))
readonly MODEL_NAME=$(eval echo $(cat ${JSON_FILE} | jq -r ".model_name"))
model="${MODEL_SAVEPATH}/${MODEL_NAME}"

readonly IMAGE_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".image_patch_size")
readonly LABEL_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".label_patch_size")
readonly OVERLAP=$(cat ${JSON_FILE} | jq -r ".overlap")
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly MASK_NAME=$(cat ${JSON_FILE} | jq -r ".mask_name")
readonly SAVE_NAME=$(cat ${JSON_FILE} | jq -r ".save_name")

# Caluculation input
readonly CSV_SAVEPATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".csv_savepath"))
readonly CLASS_LABEL=$(cat ${JSON_FILE} | jq -r ".class_label")
readonly TRUE_NAME=$(cat ${JSON_FILE} | jq -r ".true_name")
readonly PREDICT_NAME=$(cat ${JSON_FILE} | jq -r ".predict_name")




echo "---------- Training ----------"
echo "Dataset_path:${dataset_path}"
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

python3 train.py ${dataset_path} ${MODEL_SAVEPATH} ${MODULE_NAME} ${SYSTEM_NAME} ${CHECKPOINT_NAME} --train_list ${TRAIN_LIST} --val_list ${VAL_LIST} --log ${LOG} --in_channel ${IN_CHANNEL} --num_class ${NUM_CLASS} --lr ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --epoch ${EPOCH} --gpu_ids ${GPU_IDS} --api_key ${API_KEY} --project_name ${PROJECT_NAME} --experiment_name ${EXPERIMENT_NAME} 

if [ $? -ne 0 ];then
 exit 1
fi

echo "---------- Segmentation ----------"
echo ${test_list[@]}
for number in ${test_list[@]}
do
 image="${DATA_DIRECTORY}/case_${number}/${IMAGE_NAME}"
 save="${save_directory}/case_${number}/${SAVE_NAME}"

 echo "Image:${image}"
 echo "Model:${model}"
 echo "Save:${save}"
 echo "IMAGE_PATCH_SIZE:${IMAGE_PATCH_SIZE}"
 echo "LABEL_PATCH_SIZE:${LABEL_PATCH_SIZE}"
 echo "OVERLAP:${OVERLAP}"
 echo "GPU_IDS:${GPU_IDS}"


 if [ $MASK_NAME = "No" ];then
  echo "Mask:${MASK_NAME}"
  python3 segmentation.py $image $model $save --image_patch_size ${IMAGE_PATCH_SIZE} --label_patch_size ${LABEL_PATCH_SIZE} --overlap $OVERLAP -g ${GPU_IDS}

 else
  mask="${DATA_DIRECTORY}/case_${number}/${MASK_NAME}"
  echo "Mask:${mask}"

  python3 segmentation.py $image $model $save --mask_path $mask --image_patch_size ${IMAGE_PATCH_SIZE} --label_patch_size ${LABEL_PATCH_SIZE} --overlap $OVERLAP -g ${GPU_IDS}

 fi

 if [ $? -ne 0 ];then
  exit 1
 fi

done

echo "---------- Caluculation ----------"
echo "TRUE_DIRECTORY:${DATA_DIRECTORY}"
echo "PREDICT_DIRECTORY:${save_directory}"
echo "CSV_SAVEPATH:${CSV_SAVEPATH}"
echo "Test_list:${test_list[@]}"
echo "NUM_CLASS:${CLASSES}"
echo "CLASS_LABEL:${CLASS_LABEL}"
echo "TRUE_NAME:${TRUE_NAME}"
echo "PREDICT_NAME:${PREDICT_NAME}"


python3 caluculateDICE.py ${DATA_DIRECTORY} ${save_directory} ${CSV_SAVEPATH} ${TEST_LIST} --classes ${NUM_CLASS} --class_label ${CLASS_LABEL} --true_name ${TRUE_NAME} --predict_name ${PREDICT_NAME} 


