# python /project/ev_sdk/src/ji.py

import json
import os
import os.path as osp
from tkinter.messagebox import NO
import mmcv

from mmocr.apis import init_detector, model_inference
from mmocr.utils import list_from_file, list_to_file

def gen_target_path(target_root_path, src_name, suffix):
    """Gen target file path.

    Args:
        target_root_path (str): The target root path.
        src_name (str): The source file name.
        suffix (str): The suffix of target file.
    """
    assert isinstance(target_root_path, str)
    assert isinstance(src_name, str)
    assert isinstance(suffix, str)

    file_name = osp.split(src_name)[-1]
    name = osp.splitext(file_name)[0]
    return osp.join(target_root_path, name + suffix)


def save_results(result, out_dir, img_name, score_thr=0.3):
    """Save result of detected bounding boxes (quadrangle or polygon) to txt
    file.

    Args:
        result (dict): Text Detection result for one image.
        img_name (str): Image file name.
        out_dir (str): Dir of txt files to save detected results.
        score_thr (float, optional): Score threshold to filter bboxes.
    """
    assert 'boundary_result' in result
    assert score_thr > 0 and score_thr < 1

    txt_file = gen_target_path(out_dir, img_name, '.txt')
    valid_boundary_res = [
        res for res in result['boundary_result'] if res[-1] > score_thr
    ]
    lines = [
        ','.join([str(round(x)) for x in row[:-1]]) + "," + str(row[-1]) for row in valid_boundary_res
    ]
    list_to_file(txt_file, lines)

'''ev_sdk输出json样例
{
"model_data": {
    "objects": [
        {
            "points": [
                294,
                98,
                335,
                91,
                339,
                115,
                298,
                122
            ],
            "name": "店面招牌",
            "confidence": 0.9977550506591797
        },
        {
            "points": [
                237,
                111,
                277,
                106,
                280,
                127,
                240,
                133
            ],
            "name": "文字识别",
            "confidence": 0.9727065563201904
        }
    ]
    }
}
'''

def init(config, checkpoint, device="cpu"):  # 模型初始化
    # model = Pipeline()
    model = init_detector(config, checkpoint, device=device)
    if hasattr(model, 'module'):
        model = model.module

    return model


def process_image(net, input_path, out_dir=None):
    # dets = net(input_image)

    # result 格式为 [bboxes, filename], 
    # 其中bboxes是一个list，每个item的格式为 [x1,y1,x2,y2,x3,y3,x4,y4,confidence]
    result = model_inference(net, input_path)

    # ========== 输出到文件 ==========
    if out_dir is not None:
        img_name = os.path.basename(input_path)
        out_vis_dir = os.path.join(out_dir, 'out_vis_dir')
        mmcv.mkdir_or_exist(out_vis_dir)
        out_txt_dir = os.path.join(out_dir, 'out_txt_dir')
        mmcv.mkdir_or_exist(out_txt_dir)
        save_results(result, out_txt_dir, img_name, score_thr=0.5)
        # show result
        out_file = osp.join(out_vis_dir, img_name)
        kwargs_dict = {
            'score_thr': 0.5,
            'show': False,
            'out_file': out_file
        }
        net.show_result(input_path, result, **kwargs_dict)

        # print(f'\nInference done, and results saved in {out_dir}\n')


    # 包装成一个字典返回
    '''
        此示例 dets 数据为
        dets = [
            [294, 98, 335, 91, 339, 115, 298, 122, 0.9827, '店面招牌'], 
            [237, 111, 277, 106, 280, 127, 240, 133, 0.8765, '文字识别']
        ]
    '''
    out_json = {"model_data": {"objects": [], "input_path": input_path}}
    
    bboxes = []
    for bbox in result["boundary_result"]:
        points = bbox[:8]
        confidence = bbox[-1]
        text = ""  # TODO, bbox对应的文字，即 招牌
        
        single_bbox = {
            "points":points,
            "conf": confidence,
            "name": text,
        }
        out_json['model_data']['objects'].append(single_bbox)

    return json.dumps(out_json)


if __name__ == "__main__":
    config = "/Users/liuzhian/PycharmProjects/mmocr/configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py"
    checkpoint = "/Users/liuzhian/PycharmProjects/mmocr/checkpoints/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth"
    device = "cpu"
    net = init(config=config, checkpoint=checkpoint, device=device)

    input_image = '/Users/liuzhian/PycharmProjects/mmocr/demo/011515_1507187602654705.jpg'
    output = process_image(net, input_image, out_dir="./results")
    print(json.loads(output))
