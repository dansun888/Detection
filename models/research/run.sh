#!/bin/bash
# 查找脚本所在路径，并进入
#DIR="$( cd "$( dirname "$0"  )" && pwd  )"
DIR=$PWD
cd $DIR
echo current dir is $PWD

# 设置目录，避免module找不到的问题
export PYTHONPATH=$PYTHONPATH:$DIR:$DIR/slim:$DIR/object_detection

# 定义各目录
output_dir=/output  # 训练目录
dataset_dir=/data/dansun888/ssd-mobilenet-detection # 数据集目录，这里是写死的，记得修改

train_dir=$output_dir/train
checkpoint_dir=/data/dansun888/ssd-mobilenet-detection/model.ckpt
eval_dir=$output_dir/eval

# config文件
config=ssd_mobilenet_v1_pets_online.config
pipeline_config_path=/output/$config

# 先清空输出目录，本地运行会有效果，tinymind上运行这一行没有任何效果
# tinymind已经支持引用上一次的运行结果，这一行需要删掉，不然会出现上一次的运行结果被清空的状况。
# rm -rvf $output_dir/*

# 因为dataset里面的东西是不允许修改的，所以这里要把config文件复制一份到输出目录
cp /tinysrc/data/ssd_mobilenet_v1_pets_online.config /output

echo "############ training #################"
python /tinysrc/models/research/object_detection/train.py --train_dir=$output_dir/train --pipeline_config_path=$pipeline_config_path
echo "############ evaluating, this takes a while #################"
python /tinysrc/models/research/object_detection/eval.py --checkpoint_dir=$checkpoint_dir --eval_dir=$output_dir/eval --pipeline_config_path=$pipeline_config_path

# 导出模型
python /tinysrc/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path $pipeline_config_path --trained_checkpoint_prefix $train_dir/model.ckpt-500  --output_directory $output_dir/exported_graphs

# 在test.jpg上验证导出的模型
python /tinysrc/models/research/inference.py --output_dir=$output_dir --dataset_dir=$dataset_dir
