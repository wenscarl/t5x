#! /bin/bash

set -x

TFDS_DATA_DIR="/piledata/"
T5X_DIR="/workspace/rosetta-t5x-mirror/"

# Arguments
OPTIMIZER=$1
T5_SIZE=$2       # Model size (small, base, large)
PREC="$3"        # Precision (float32, float16, bfloat16)
GPUS_PER_NODE=$4      # Number of GPUs (1, 2, 4, 8)
BSIZE_PER_GPU=$5 # Batch size per GPU (varies with model size)
ENABLE_XLAFP8=${6:-0} # Whether to enable XLA native fp8 support (0, 1)
LOG_DIR=$7       # Output log directory
MODEL_DIR_LOCAL=${8:-"benchmark_pile_dump"}
MODEL_DIR=${T5X_WORKSPACE_DIR}/${MODEL_DIR_LOCAL}
NUM_MICROBATCHES=${9:-0}
MP=${10:-1}

echo Model Parallel partitions: ${MP}

STEPS=400

# optimizer checking
case $OPTIMIZER in
  adam)
    ;;
  adafactor)
    ;;
  adam_nqs_as)
    ;;
  adam_qs_as)
    ;;
  *)
    echo $OPTIMIZER optimizer not supported. Try adam, adam_nqs_as, adam_qs_as, or adafactor.
    exit
esac

echo Model and training stats will be saved to: $MODEL_DIRlar

if [ -z $CUDA_VISIBLE_DEVICES ]; then
case $GPUS_PER_NODE in

  1)
    export CUDA_VISIBLE_DEVICES="0"
    ;;

  2)
    export CUDA_VISIBLE_DEVICES="0,1"
    ;;

  4)
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    ;;

  8)
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    ;;

  *)
    echo "${NUM_GPUS} not supported"
    exit
esac
fi

# Global batch size
GLOBAL_BATCH_SIZE=$(( GPUS_PER_NODE * BSIZE_PER_GPU))

rm -rf ${T5X_WORKSPACE_DIR}/model_dir/checkpoint* \
rm -rf ${MODEL_DIR}/*
#  --gin_file="${T5X_DIR}/t5x/examples/t5/t5_1_1/examples/${T5_SIZE}_pile_pretrain_${OPTIMIZER}.gin" \
export ENABLE_TE=0

#CUDA_VISIBLE_DEVICES=0 TF_DUMP_GRAPH_PREFIX=/tmp/generated TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2" XLA_FLAGS="--xla_dump_hlo_as_html --xla_dump_to=/tmp/generated --xla_gpu_graph_level=0 --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_reduction_epilogue_fusion=false --xla_dump_hlo_pass_re=.*" python3 ${T%X_DIR}/t5x/train.py \

#CUDA_VISIBLE_DEVICES=0 XLA_FLAGS="--xla_gpu_graph_level=0 --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_reduction_epilogue_fusion=false" python3 ${T5X_DIR}/t5x/train.py \
#CUDA_VISIBLE_DEVICES=0 TF_DUMP_GRAPH_PREFIX=/tmp/generated TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2" XLA_FLAGS="--xla_dump_to=/tmp/generated --xla_dump_hlo_as_text --xla_gpu_graph_level=0 --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_reduction_epilogue_fusion=false --xla_dump_hlo_pass_re=.*" python3 ${T5X_DIR}/t5x/train.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_FLAGS="--xla_gpu_graph_level=0 --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_reduction_epilogue_fusion=false" python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="${T5X_DIR}/t5x/examples/decoder_only/examples/base_wmt_from_scratch.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=${STEPS}\
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
  --gin.partitioning.PjitPartitioner.num_partitions=${MP} \
  --gin.train/utils.DatasetConfig.batch_size=${GLOBAL_BATCH_SIZE} \
  --gin.train_eval/utils.DatasetConfig.batch_size=${GLOBAL_BATCH_SIZE} \
  --gin.infer_eval/utils.DatasetConfig.batch_size=${GLOBAL_BATCH_SIZE} \
  --gin.xla_fp8_helper.XLAFp8Config.enabled=${ENABLE_XLAFP8} \
  --gin.train.stats_period=200

set +x
