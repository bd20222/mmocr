# Text Recognition Testing set, including:
# Regular Datasets: IIIT5K, SVT, IC13
# Irregular Datasets: IC15, SVTP, CT80

train_root = '/Users/liuzhian/datasets/svt'

train_img_prefix = f'{train_root}/svt1/'
train_ann_file = f'{train_root}/svt1/test_label.txt'

train = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix,
    ann_file=train_ann_file,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)


train_list = [train]
