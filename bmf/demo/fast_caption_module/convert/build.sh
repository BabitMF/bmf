#!/bin/bash
set -e

build_vit() {
  SRC=$1
  TARGET=$2
  echo $SRC $TARGET
  if [ ! -d $TARGET ]; then
    mkdir -p $TARGET
  fi

  NAME=$(basename $SRC)

  if [ ! -d $TARGET ]; then
    mkdir -p $TARGET
  fi

  DIR=$(dirname $TARGET)

  ONNXDIR=$DIR/ViT_$NAME
  python3 export_vit.py $SRC $ONNXDIR
  cp $ONNXDIR/image_newline.pth $TARGET/
  cp $ONNXDIR/image_newline_c++.pth $TARGET/

  ONNX=$ONNXDIR/vit.onnx
  ENGINE=$TARGET/vit.trt

  trtexec --onnx=$ONNX --fp16 --minShapes=x:1x3x336x336 \
                              --optShapes=x:40x3x336x336 \
                              --maxShapes=x:160x3x336x336 \
                              --saveEngine=$ENGINE
}

build_llm() {
  SRC=$1
  NAME=$(basename $SRC)

  TARGET=$2
  BS=$3
  echo $SRC $TARGET $BS
  if [ ! -d $TARGET ]; then
    mkdir -p $TARGET
  fi

  DIR=$(dirname $TARGET)
  echo $PYTHONPATH
  CONVERT=$DIR/Converted_$NAME
  echo $SRC
  echo $CONVERT
  python3 convert_checkpoint.py --model_dir $SRC \
                                --output_dir $CONVERT \
                                --dtype float16
                                

  MMLEN=$((BS * 6000))
    
  trtllm-build \
    --tp_size 1 \
    --pp_size 1 \
    --checkpoint_dir $CONVERT \
    --output_dir $TARGET/llm_engine \
    --gpt_attention_plugin float16 \
    --gemm_plugin float16 \
    --use_fused_mlp \
    --max_batch_size 1 \
    --max_input_len 16384 \
    --max_output_len 256 \
    --max_num_tokens 16384 \
    --max_multimodal_len 8192
      
  cp $SRC/config.json $TARGET
  cp $SRC/tokenizer* $TARGET
}

SRC=$1
CLIPSRC=$2
TARGET=$3
CODEPATH=$4
export PYTHONPATH=$PYTHONPATH:$CODEPATH
BS=2

# build_vit $SRC $TARGET
build_llm $SRC $TARGET $BS

if [[ -e $CLIPSRC/preprocessor_config.json ]]
then
cp $CLIPSRC/preprocessor_config.json $TARGET/
else
printf "{
  \"crop_size\": 336,
  \"do_center_crop\": true,
  \"do_normalize\": true,
  \"do_resize\": true,
  \"feature_extractor_type\": \"CLIPFeatureExtractor\",
  \"image_mean\": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  \"image_std\": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  \"resample\": 3,
  \"size\": 336
}" > $TARGET/preprocessor_config.json
fi

