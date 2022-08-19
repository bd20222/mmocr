from time import sleep
import cv2
import numpy as np
import os 
import xml.sax
from PIL import Image, ImageDraw, ImageFont


class TicketHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.areas = []
    
    # 元素开始事件处理
    def startElement(self, tag, attributes):
        if tag == "root":
            zoom = attributes["zoom"]
        elif tag == "area":
            index = int(attributes["index"])
            points = [tuple(map(float, p[1:-1].split(','))) for p in attributes["points"].split(';')[:-1]]
            
            x1, y1 = min([p[0] for p in points]), min([p[1] for p in points])
            x2, y2 = max([p[0] for p in points]), max([p[1] for p in points])
            bbox = [x1, y1, x2, y2]
            
            pointnum = int(attributes["pointnum"])
            assert len(points) == pointnum, "Pointnum and the given points are not consistent!"
            text = attributes["text"]
            type = attributes["type"]

            self.areas.append({"index":index, "points":points, "bbox":bbox, "text":text, "type":type})
    
    # 元素结束事件处理
    def endElement(self, tag):
        if tag == "root":
            print("Handling area done, counting (%d) area in total!"%len(self.areas))
            
        elif tag == "area":
            # print("...")
            pass

        
def vis_image_anno(sample_name):
    """ 可视化每张图片的 annotation """
    XML_file = "tests/data/tiny_medical_tickets/%s.xml"%sample_name
    img_file = "tests/data/tiny_medical_tickets/%s.jpg"%sample_name

    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    
    Handler = TicketHandler()
    parser.setContentHandler(Handler)
    
    parser.parse(XML_file)
    areas = parser._cont_handler.areas

    img = Image.open(img_file).convert("RGB")
    img_canvas = ImageDraw.Draw(img)
    

    font_path = "/Library/Fonts/Artifakt Element Bold.ttf"
    font = ImageFont.truetype(font=font_path, size=30)
    for area in areas:
        img_canvas.polygon(area["points"], width=10, outline="blue")
        img_canvas.text((area["bbox"][0]-30, area["bbox"][1]-30), text = "%s"%area["type"], font=font)
    
    img.save("tests/data/tiny_medical_tickets/%s-vis.png"%sample_name)


def vis_icdar2015_sample(img_file, anno_file):
    img = Image.open(img_file).convert("RGB")
    img_canvas = ImageDraw.Draw(img)
    
    with open(anno_file, "r", encoding='utf-8-sig') as f:
        lines = [l.strip() for l in f.readlines()]

    for l in lines:
        points_xy = list(map(int, l.split(',')[:-1]))
        text = l.split(',')[-1]

        if text != "###":
            img_canvas.polygon(points_xy, width=4, outline="blue")
        else:
            img_canvas.polygon(points_xy, width=4, outline="red")
        
    img.save("icdar2015-vis.png")

if __name__ == "__main__":
    # # 可视化发票数据集
    # for i in range(5,6):
    #     vis_image_anno(str(i+1))
   

    # 可视化icdar 2015数据集
    vis_icdar2015_sample(img_file="/Users/liuzhian/PycharmProjects/mmocr/tests/data/icdar2015/imgs/training/img_371.jpg",
                        anno_file="/Users/liuzhian/PycharmProjects/mmocr/tests/data/icdar2015/annotations/training/gt_img_371.txt")

         

