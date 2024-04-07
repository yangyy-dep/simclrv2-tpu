#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
#这一行获取当前脚本所在的目录，并将其赋值给变量 model_dir。dirname 函数用于获取给定路径的目录名，readlink -f 用于获取指定文件的绝对路径
#返回当前脚本所在的目录的绝对路径
if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi
#如何传参bash script.sh arg1 arg2 ...，$1 获取的第一个参数的值
outdir=../models/BM1684X
#$1 是一个占位符，表示第一个传入的参数
function gen_mlir()
{
    model_transform.py \
        --model_name simclrv2_$1b \
        --model_def simclrv2_$1b.onnx \
        --input_shapes [[$1,3,32,32]] \
        --keep_aspect_ratio \
        --pixel_format rgb  \
        --mlir simclrv2_$1b.mlir
}

function gen_fp16bmodel()
{
    model_deploy.py \
        --mlir simclrv2_$1b.mlir \
        --quantize F16 \
        --chip $target \
        --model simclrv2_fp16_$1b.bmodel

    mv simclrv2_fp16_$1b.bmodel $outdir/
}
#进入脚本所在目录，检查输出目录是否存在
pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
#fi表示if语句块的结束
# batch_size=1
#调用之前定义的 gen_mlir() 和 gen_fp16bmodel() 函数，传入参数 64
gen_mlir 1
gen_fp16bmodel 1
#返回原始的工作目录
popd
