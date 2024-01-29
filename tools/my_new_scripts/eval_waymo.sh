DATA_SRC=${1}
MODEL_CFG_PATH=${2}
LOG_FILE=${3}
EXTRA_TAG=${4}
MODEL_PTH_PATH=${5}
WORKING_DIR=${6}
batch_size=${7}
NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

cd ${WORKING_DIR}


rm ../data/waymo/waymo_processed_data
ln -s ${DATA_SRC} ../data/waymo/waymo_processed_data

bash ./scripts/dist_test.sh ${NUM_GPU} ${LOG_FILE} \
    --cfg_file ${MODEL_CFG_PATH} \
    --batch_size ${batch_size} \
    --ckpt ${MODEL_PTH_PATH} \
    --extra_tag ${EXTRA_TAG} \
