# python /project/ev_sdk/src/ji.py

import json
import os
import os.path as osp
from PIL import Image
import mmcv
import numpy as np
from signboard_infer.ocr_pipiline import OCR_Pipeline

from mmocr.apis import init_detector, model_inference
from mmocr.utils import list_from_file, list_to_file
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.core.visualize import det_recog_show_result

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

def init(det_config, det_checkpoint,
         rcg_config, rcg_checkpoint, device="cpu"):  # 模型初始化
    
    det_model = init_detector(det_config, det_checkpoint, device=device)
    if hasattr(det_model, 'module'):
        det_model = det_model.module

    rcg_model = init_detector(rcg_config, rcg_checkpoint, device=device)
    if hasattr(rcg_model, 'module'):
        rcg_model = rcg_model.module

    # build the overall model, i.e., detector + recognizer
    model = OCR_Pipeline(det_model=det_model, rcg_model=rcg_model)
    
    return model


def process_image(net, input_path, out_dir=None):

    ocr_res, ocr_res_vis_img = net.processing_img(input_path=input_path, pp_visulize = False)

    # ========== 输出到文件 ==========
    if out_dir is not None:
        assert ocr_res_vis_img is not None, "pp_visulize should be specified as True."
        img_name = os.path.basename(input_path)
        out_vis_dir = os.path.join(out_dir, 'out_vis_dir')
        mmcv.mkdir_or_exist(out_vis_dir)
        mmcv.imwrite(ocr_res_vis_img, osp.join(out_dir, img_name))
        # print(f'\nInference done, and results saved in {out_dir}\n')

    # 包装成一个字典返回
    '''
        此示例 dets 数据为
        dets = [
            [294, 98, 335, 91, 339, 115, 298, 122, 0.9827, '店面招牌'],   # (x1,y1) 为左上角，四个点顺时针
            [237, 111, 277, 106, 280, 127, 240, 133, 0.8765, '文字识别']
        ]
    '''
    out_json = {"model_data": {"objects": [], "input_path": input_path}}
    
    # result 格式为 [bboxes, filename], 
    # 其中bboxes是一个list，每个item的格式为 [x1,y1,x2,y2,x3,y3,x4,y4,confidence]
    for bbox_dict in ocr_res["result"]:
        points = bbox_dict["box"]
        confidence = bbox_dict["box_score"]
        text = bbox_dict["text"] 
        text_score = bbox_dict["text_score"]  # 暂时没用到
        
        single_bbox = {
            "points":points,
            "conf": confidence,
            "name": text,
        }
        out_json['model_data']['objects'].append(single_bbox)

    return json.dumps(out_json)


if __name__ == "__main__":

    det_config = "/home/liuzhian/hdd4T/code/mmocr/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py"
    det_checkpoint = "/home/liuzhian/.cache/torch/hub/checkpoints/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth"
    # det_config = "/home/liuzhian/hdd4T/code/mmocr/configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py"
    # det_checkpoint = "/home/liuzhian/hdd4T/code/mmocr/checkpoints/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth"
    device = "cuda:0"

    rcg_config = "/home/liuzhian/hdd4T/code/mmocr/configs/textrecog/seg/seg_r31_1by16_fpnocr_academic.py"
    rcg_checkpoint = "/home/liuzhian/.cache/torch/hub/checkpoints/seg_r31_1by16_fpnocr_academic-72235b11.pth"
    # rcg_config = "/home/liuzhian/hdd4T/code/mmocr/configs/textrecog/abinet/abinet_vision_only_academic.py"
    # rcg_checkpoint = "/home/liuzhian/hdd4T/code/mmocr/checkpoints/textrecog/abinet/abinet_vision_only_academic-e6b9ea89.pth"
    
    
    net = init(det_config=det_config, det_checkpoint=det_checkpoint,
               rcg_config=rcg_config, rcg_checkpoint=rcg_checkpoint,
               device=device)

    # input_image = '/Users/liuzhian/PycharmProjects/mmocr/demo/011515_1507187602654705.jpg'
    # input_image = np.array(Image.open(input_image).convert('RGB'))
    input_image = "/home/liuzhian/hdd4T/code/mmocr/demo/demo_text_ocr.jpg"
    output = process_image(net, input_image)
    print(json.loads(output))
