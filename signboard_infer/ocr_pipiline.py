
import json
import os
import os.path as osp
from PIL import Image
import mmcv
import numpy as np

from mmocr.apis import init_detector, model_inference
from mmocr.utils import list_from_file, list_to_file
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.core.visualize import det_recog_show_result

class OCR_Pipeline:
    def __init__(self, det_model=None, rcg_model=None):
        self.det_model = det_model
        self.rcg_model = rcg_model
    
    def processing_img(self, input_path, rcg_batch_mode=False, pp_visulize=False):
        assert not rcg_batch_mode, "目前还不支持 text recognition model 批处理!"

        arr = mmcv.imread(input_path)

        # text detection
        det_result = model_inference(self.det_model, arr, batch_mode=False)

        bboxes = det_result['boundary_result']

        # For each bounding box, the image is cropped and
        # sent to the recognition model either one by one
        # or all together depending on the rcg_batch_mode        
        img_e2e_res = {}
        img_e2e_res['filename'] = input_path
        img_e2e_res['result'] = []
        
        box_imgs = []
        for bbox in bboxes:
            box_res = {}
            box_res['box'] = [round(x) for x in bbox[:-1]]
            box_res['box_score'] = float(bbox[-1])
            box = bbox[:8]
          
            box_img = crop_img(arr, box)
            if rcg_batch_mode:
                box_imgs.append(box_img)
            else:
                recog_result = model_inference(self.rcg_model, box_img)
                text = recog_result['text']
                text_score = recog_result['score']
                if isinstance(text_score, list):
                    text_score = sum(text_score) / max(1, len(text))
                box_res['text'] = text
                box_res['text_score'] = text_score
            img_e2e_res['result'].append(box_res)
        
        # TODO: inference the text recognition model all at once
        if rcg_batch_mode:
            recog_results = model_inference(self.rcg_model, box_imgs, batch_mode=True)
            for i, recog_result in enumerate(recog_results):
                text = recog_result['text']
                text_score = recog_result['score']
                if isinstance(text_score, (list, tuple)):
                    text_score = sum(text_score) / max(1, len(text))
                img_e2e_res['result'][i]['text'] = text
                img_e2e_res['result'][i]['text_score'] = text_score
        
        if pp_visulize:
            res_img = det_recog_show_result(arr, img_e2e_res, out_file=None)
        else:
            res_img = None

        return img_e2e_res, res_img
  