"""将原始的广告牌数据集转换为COCO格式的数据集"""
import os
import random
import glob
import shutil
import argparse

def split_train_test(root_path, out_path, train_frac=0.85):
    imgs = sorted(glob.glob(os.path.join(root_path, "*.jpg")))
    txts = sorted(glob.glob(os.path.join(root_path, "*.txt")))

    for img, txt in zip(imgs, txts):
        assert os.path.basename(img).split(".")[0] == \
                os.path.basename(txt).split(".")[0] , "Image and annotation are inconsistent!"

    N = len(imgs)
    N_trian = int(N*train_frac)
    indices = list(range(N))
    random.shuffle(indices)

    indices_train, indices_test = indices[:N_trian], indices[N_trian:]

    indices_splits = [
        ["training", indices_train],
        ["test", indices_test]
    ]

    # copy/mv the dataset
    os.makedirs(os.path.join(out_path, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "imgs"), exist_ok=True)

    for split in indices_splits:
        os.makedirs(os.path.join(out_path, "annotations", split[0]), exist_ok=True)
        os.makedirs(os.path.join(out_path, "imgs", split[0]), exist_ok=True)

        for idx in split[1]:
            img_i, txt_i = imgs[idx], txts[idx]

            shutil.copy(img_i, os.path.join(out_path, "imgs", split[0], os.path.basename(img_i)))
            shutil.copy(txt_i, os.path.join(out_path, "annotations", split[0], os.path.basename(txt_i)))
    
    print("# Training set: %d, # test set: %d."%(N_trian, N - N_trian))


def parse_args():
    parser = argparse.ArgumentParser(description='split original signboard dataset into COCO format')
    parser.add_argument('--original_ds_root', 
        default="/Users/liuzhian/PycharmProjects/mmocr/tests/data/sample_data/1413",
        help='original signboard dataset root path')
    parser.add_argument('--out_ds_root',
        default="/Users/liuzhian/PycharmProjects/mmocr/tests/data/sample_data/1413_coco",
        help='output signboard dataset root path')
    parser.add_argument('--train_frac',
        type=float,
        default=0.85,
        help='training set fraction')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    split_train_test(root_path=args.original_ds_root,
                    out_path=args.out_ds_root,
                    train_frac=args.train_frac)


if __name__ == "__main__":
    main()