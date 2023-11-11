DATA_SRC=${1}
non_detected_obj_dir=${2}
AP_files_dir=${3}
MODEL_CFG_PATH=${4}
EPOCH=${5}
EXTRA_TAG=${6}
WORKING_DIR=${7}
batch_size=${8}
LOG_PATH=${9}
NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


cd ${WORKING_DIR}

mkdir ../pcdet_object_analysis/
mkdir ../AP_files/
mkdir -p ${non_detected_obj_dir}
mkdir -p ${AP_files_dir}


rm ../data/kitti/training/velodyne
ln -s ${DATA_SRC} ../data/kitti/training/velodyne
ln -s /pcc-storage/KITTI/training/planes ../data/kitti/training/planes

bash scripts/dist_train.sh ${NUM_GPU} ${LOG_PATH} \
    --cfg_file ${MODEL_CFG_PATH} \
    --epochs ${EPOCH} \
    --extra_tag ${EXTRA_TAG} \
    --num_epochs_to_eval 30 \
    --batch_size ${batch_size} \