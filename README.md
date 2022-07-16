# Anaconda Environment Install 环境安装

    $ conda env create -f swin_env.yaml
同时需要安装编译Nvidia Apex组件
# Data preparation
使用标准ImageNet数据集结构，对于标准文件夹数据集结构如下：

    $ tree data
    dataset
    ├── train
    │   ├── class1
    │   │   ├── img1.jpeg
    │   │   ├── img2.jpeg
    │   │   └── ...
    │   ├── class2
    │   │   ├── img3.jpeg
    │   │   └── ...
    │   └── ...
    └── val
        ├── class1
        │   ├── img4.jpeg
        │   ├── img5.jpeg
        │   └── ...
        ├── class2
        │   ├── img6.jpeg
        │   └── ...
        └── ...


# Train 训练
训练命令

    $ python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg "config path" --pretrained "pretrained_model path" --data-path "dataset path" --batch-size "batch_size" --accumulation-steps 8 [--use-checkpoint]

如选用配置文件configs/mytest_0703.yaml，选用预训练模型swin_base_patch4_window7_224.pth，以及数据集路径../dataset_0703/added_data/，则训练命令为

    $ python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg configs/mytest_0703.yaml --pretrained swin_base_patch4_window7_224.pth --data-path ../dataset_0703/added_data/ --batch-size 32 --accumulation-steps 8 [--use-checkpoint]

# Test 测试

    $ python estimate.py
在 *estimate.py* 文件中可以更改 line64:

    check = "output/ckpt_epoch_0712.pth"
为checkpoint路径。
更改line70:

    image_path = "../dataset_0703/added_data/test"
为测试集路径

# 其他配置更改
可在 *config.py* 中对参数进行修改，或在运行python main.py时增加或修改相关参数。

