#需要使用量化表simclr2.qtable
#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi

outdir=../models/BM1684X

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

function gen_cali_table()
{
    run_calibration.py rsimclrv2_$1b.mlir \
        --dataset ../datasets/test_data/ \
        --input_num 100 \
        -o simclrv2_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir simclrv2_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table simclrv2_cali_table \
        --model simclrv2_int8_$1b.bmodel
        --compare_all 
        --debug 
        --quantize_table ../simclr2.qtable

    mv simclrv2_int8_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1



popd