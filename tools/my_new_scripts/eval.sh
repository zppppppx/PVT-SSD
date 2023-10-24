DATA_SRC=${1}
non_detected_obj_dir=${2}
AP_files_dir=${3}
MODEL_CFG_PATH=${4}
EPOCH=${5}
EXTRA_TAG=${6}
MODEL_PTH_PATH=${7}
WORKING_DIR=${8}
batch_size=${9}
NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

cd ${WORKING_DIR}

mkdir ../pcdet_object_analysis/
mkdir ../AP_files/
mkdir -p ${non_detected_obj_dir}
mkdir -p ${AP_files_dir}

rm ../data/kitti/training/velodyne
ln -s ${DATA_SRC} ../data/kitti/training/velodyne

bash ./scripts/dist_test.sh ${NUM_GPU} \
    --cfg_file ${MODEL_CFG_PATH} \
    --batch_size ${batch_size} \
    --ckpt ${MODEL_PTH_PATH} \
    --extra_tag ${EXTRA_TAG} \


cp ../pcdet_object_analysis/*npy ${non_detected_obj_dir}
rm ../pcdet_object_analysis/*npy

cp ../AP_files/*npy ${AP_files_dir}
rm ../AP_files/*npy