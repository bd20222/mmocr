# Text Recognition for signboard dataset

data_root = '/Users/liuzhian/datasets/sample_data'

train = dict(
    type='OCRDataset',
    img_prefix=f'{data_root}/1414_text_recog/',
    ann_file=f'{data_root}/1414_text_recog/label_training.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='###')),
    pipeline=None,
    test_mode=False)

test = dict(
    type='OCRDataset',
    img_prefix=f'{data_root}/1414_text_recog/',
    ann_file=f'{data_root}/1414_text_recog/label_test.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='###')),
    pipeline=None,
    test_mode=False)

train_list = [train]
test_list = [test]
