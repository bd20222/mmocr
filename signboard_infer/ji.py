# python /project/ev_sdk/src/ji.py

import json
import os

from mmocr.apis import init_detector, model_inference


'''
ev_sdk输出json样例
{
"011284_1508290961970980.jpg": {
    "bboxes": [
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
            "text": "店面招牌",
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
            "text": "文字识别",
            "confidence": 0.9727065563201904
        }
    ]
    "filename": "/home/data/1413/011284_1508290961970980.jpg" 
    }
}
'''

def init(config, checkpoint, device="cpu"):  # 模型初始化
    # model = Pipeline()
    model = init_detector(config, checkpoint, device=device)
    if hasattr(model, 'module'):
        model = model.module

    return model


def process_image(net, input_path, args=None):
    # dets = net(input_image)

    # result 格式为 [bboxes, filename], 
    # 其中bboxes是一个list，每个item的格式为 [x1,y1,x2,y2,x3,y3,x4,y4,confidence]
    result = model_inference(net, input_path)

    '''
        此示例 dets 数据为
        dets = [
            [294, 98, 335, 91, 339, 115, 298, 122, 0.9827, '店面招牌'], 
            [237, 111, 277, 106, 280, 127, 240, 133, 0.8765, '文字识别']
        ]
    '''
    model_pred = {os.path.basename(input_path): {}}
    
    bboxes = []
    for bbox in result["boundary_result"]:
        points = bbox[:8]
        confidence = bbox[-1]
        text = ""  # TODO, bbox对应的文字，即 招牌
        
        single_bbox = {
            "points":points,
            "conf": confidence,
            "text": text,
        }
        bboxes.append(single_bbox)

    model_pred[os.path.basename(input_path)]["filename"] = input_path
    model_pred[os.path.basename(input_path)]["bboxes"] = bboxes

    return json.dumps(model_pred)


if __name__ == "__main__":
    config = ""
    checkpoint = ""
    device = ""
    net = init(config=config, checkpoint=checkpoint, device=device)
    
    input_image = '/home/data/1413/011284_1508290961970980.jpg'
    output = process_image(net, input_image)
    print(json.loads(output))
