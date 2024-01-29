# #!/usr/bin/env bash

# while true
# do
#     PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done
# echo $PORT



# NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# EPOCH=epoch_30

# CFG_NAME=kitti_models/pvt_ssd
# TAG_NAME=${1}

# CKPT=${2}

# LOG_FILE=${3}

# mkdir ../AP_files/

# python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port $PORT test.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 2 --extra_tag $TAG_NAME --ckpt $CKPT \
#     > ${LOG_FILE} 2>&1

# # python3 test.py --cfg_file cfgs/$CFG_NAME.yaml --extra_tag $TAG_NAME --ckpt $CKPT \
# #     > ${LOG_FILE} 2>&1

# # GT=../data/waymo/gt.bin
# # EVAL=../data/waymo/compute_detection_metrics_main
# # DT_DIR=../output/$CFG_NAME/$TAG_NAME/eval/$EPOCH/val/default/final_result/data

# # $EVAL $DT_DIR/detection_pred.bin $GT


#!/usr/bin/env bash

set -x
NGPUS=$1
LOG_FILE=$2
PY_ARGS=${@:3}

python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch ${PY_ARGS} > $LOG_FILE 2>&1