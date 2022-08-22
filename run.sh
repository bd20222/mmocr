# 删除掉已经存在的数据集
# rm -rf /home/data/1413_coco

cd /project/train/src_repo

# 将原始的数据集转化为COCO的格式
python3 mmocr/ds_signboard2coco.py \
        --original_ds_root=/home/data/1413 \
        --out_ds_root=/home/data/1413_coco

# 再制作好对应的 annotaiton 文件 (json格式), 训练集 和 测试集
python3 mmocr/tools/data/textdet/signboard_converter.py \
        --signboard_path=/home/data/1413_coco

# 开始训练
python3 mmocr/tools/train.py mmocr/configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_signboard.py \
        --work-dir=../models
    