
# Set dataset path      
export DETECTRON2_DATASETS=/data2/objdet/
export CUDA_VISIBLE_DEVICES=0,1,2,3
[[ -z "$RANK" ]] && RANK=0
[[ -z "$AZUREML_NODE_COUNT" ]] && NODE_COUNT=1 || NODE_COUNT=$AZUREML_NODE_COUNT
[[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_ADDR=127.0.0.1 || MASTER_ADDR=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 1)
[[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_PORT=29500 || MASTER_PORT=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 2)

# Set the number of GPUs for training
GPUS=4

echo "rank: ${RANK}"
echo "node count: ${NODE_COUNT}"
echo "master addr: ${MASTER_ADDR}"
echo "master port: ${MASTER_PORT}"
echo "num gpus: ${GPUS}"
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export OMP_NUM_THREADS=4

python train_net.py  \
        --machine-rank ${RANK} \
        --num-machines ${NODE_COUNT}\
        --num-gpus ${GPUS} \
	"$@"
