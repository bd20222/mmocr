# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET
import mmcv
import random
import cv2
import glob
import shutil

from mmocr.utils.fileio import list_to_file
from mmocr.utils import list_from_file

def split_train_test(all_labels_file, ds_root_path, train_frac=0.85):
    
    lines = list_from_file(all_labels_file, encoding='utf-8-sig')
    
    # 分割训练集、测试集
    N = len(lines)
    N_trian = int(N*train_frac)
    indices = list(range(N))
    random.shuffle(indices)
    indices_train, indices_test = indices[:N_trian], indices[N_trian:]
    indices_splits = [
        ["training", indices_train],
        ["test", indices_test]
    ]

    for split in indices_splits:
        os.makedirs(os.path.join(ds_root_path, "image_%s"%split[0]), exist_ok=True)

        lines_split = []
        for idx in split[1]:
            line = lines[idx]
            img_path_origin, text = line.split("###")
            img_path_new = osp.join("image_%s"%split[0], osp.basename(img_path_origin))

            lines_split.append("%s###%s"%(img_path_new, text))

            shutil.copy(osp.join(ds_root_path, img_path_origin),
                        osp.join(ds_root_path, img_path_new))

        list_to_file(os.path.join(ds_root_path, "label_%s.txt"%split[0]), lines_split)

    print("# Training set: %d, # test set: %d."%(N_trian, N - N_trian))
 

def cnt_text_max_len(text_label_file):
    # 统计最长文本长度
    max_len = 0 

    lines = list_from_file(text_label_file, encoding='utf-8-sig')
    for line in lines:
        text = line.split(" ")[-1]
        max_len = max(max_len, len(text))

    return max_len
    

def build_char_dict(text_label_file, char_dict_file):
    DICT91='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]_`~ ' # 最后包含了空格

    # 再统计中文
    new_chars = []
    char_set = set(DICT91)

    lines = list_from_file(text_label_file, encoding='utf-8-sig')
    for line in lines:
        text = line.split(" ")[-1]
        for ch in text:
            if ch not in char_set:
                char_set.add(ch)
                new_chars.append(ch)

    new_chars.sort()
    new_chars_str = "".join(new_chars)

    all_chars = DICT91.split('') + new_chars_str
    list_to_file(char_dict_file, all_chars)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate signboard dataset by cropping box image.')
    parser.add_argument(
        '--root_path',
        help='Root dir path of original signboard, where images and annotaions are inside')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--resize',
        action='store_true',
        help='Whether resize cropped image to certain size.')
    parser.add_argument('--height', default=32, help='Resize height.')
    parser.add_argument('--width', default=100, help='Resize width.')
    parser.add_argument('--train_frac',
        type=float,
        default=0.85,
        help='training set fraction')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    out_dir = args.out_dir if args.out_dir else root_path
    mmcv.mkdir_or_exist(out_dir)

    img_files = sorted(glob.glob(osp.join(root_path, "*.jpg")))
    txt_files = sorted(glob.glob(osp.join(root_path, "*.txt")))
    for img_f, txt_f in zip(img_files, txt_files):
        assert osp.basename(img_f).split('.')[0] == osp.basename(txt_f).split('.')[0], "数据集图片和标注不匹配!"


    # outputs
    dst_label_file = osp.join(out_dir, 'label.txt')
    # dst_char_dict_file = osp.join(out_dir, 'char_dictionary.txt')
    dst_image_root = osp.join(out_dir, 'image')
    os.makedirs(dst_image_root, exist_ok=True)
    
    index = 1
    lines = []
    total_img_num = len(img_f)
    for i, (img_f, txt_f) in enumerate(zip(img_files, txt_files)):
        print(f'[{i+1}/{total_img_num}] Process image: {img_f}')
        
        src_img = cv2.imread(img_f)
        gt_list = list_from_file(txt_f, encoding='utf-8-sig')

        for line in gt_list:
            # each line has one ploygen (4 vetices), and text
            # e.g., 695,885,866,888,867,1146,696,1143,你好
            line = line.strip()
            strs = line.split(',')
            points, text_label = strs[:-1], strs[-1]
            # if len(text_label) == 0:
            #     text_label = " "  # 空格

            xs = [int(x) for x in points[0::2]] # x1,x2,x3,x4 顺时针
            ys = [int(x) for x in points[1::2]] # y1,y2,y3,y4 顺时针
    
            rb, re = max(0, min(ys)), min(src_img.shape[0], max(ys))
            cb, ce = max(0, min(xs)), min(src_img.shape[1], max(xs))
            dst_img = src_img[rb:re, cb:ce]
            
            if args.resize:
                dst_img = cv2.resize(dst_img, (args.width, args.height))
            dst_img_name = f'img_{index:04}' + '.jpg'
            index += 1
            dst_img_path = osp.join(dst_image_root, dst_img_name)
            cv2.imwrite(dst_img_path, dst_img)

            lines.append(f'{osp.basename(dst_image_root)}/{dst_img_name}###{text_label}')

    list_to_file(dst_label_file, lines)

    # # 生成字典
    # build_char_dict(dst_label_file, dst_char_dict_file)

    # 统计最大长度
    max_text_len = cnt_text_max_len(dst_label_file)

    # 生成训练集、测试集
    split_train_test(all_labels_file=dst_label_file,
                    ds_root_path=out_dir,
                    train_frac=args.train_frac)

    print(f'Finish to generate signboard text recognition dataset')


if __name__ == '__main__':
    main()

