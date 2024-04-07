#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
#��һ�л�ȡ��ǰ�ű����ڵ�Ŀ¼�������丳ֵ������ model_dir��dirname �������ڻ�ȡ����·����Ŀ¼����readlink -f ���ڻ�ȡָ���ļ��ľ���·��
#���ص�ǰ�ű����ڵ�Ŀ¼�ľ���·��
if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi
#��δ���bash script.sh arg1 arg2 ...��$1 ��ȡ�ĵ�һ��������ֵ
outdir=../models/BM1684X
#$1 ��һ��ռλ������ʾ��һ������Ĳ���
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
#����ű�����Ŀ¼��������Ŀ¼�Ƿ����
pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
#fi��ʾif����Ľ���
# batch_size=1
#����֮ǰ����� gen_mlir() �� gen_fp16bmodel() ������������� 64
gen_mlir 1
gen_fp16bmodel 1
#����ԭʼ�Ĺ���Ŀ¼
popd
