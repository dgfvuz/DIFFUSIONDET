# 样例集说明：
# Img文件夹为图像集，txt文件夹为标注文件，其中标注文件为txt格式，每一行代表标注的一个目标：5个数字分别是 class_num x y w h
# 如：0 0.521000 0.235075 0.362000 0.450249
# 第一个数表示 类别，数字0对应classes.txt中的第一个类Mouse_bite，其余4个数字表示标注框的中心坐标(x,y),标注框的相对宽和高w,h。

# 标注文件class_num和缺陷种类对应关系为：
# 0 Mouse_bite
# 1 Open_circuit
# 2 Short
# 3 Spur
# 4 Spurious_copper
import os
import os
import random
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from PIL import Image
import shutil
import cv2
import numpy as np
import multiprocessing as mp
import json
import multiprocessing as mp
import numpy as np
import os
import time
import cv2
from detectron2.config import get_cfg
from diffusiondet.predictor import VisualizationDemo
from diffusiondet import  add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs

# 设置config
def setup_cfg():
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    # 添加config文件
    cfg.merge_from_file("configs/diffdet.coco.res101.yaml")
    cfg.merge_from_list([])
    # 设置阈值
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    return cfg

# 文件夹路径, 这是源文件夹
# 源图片文件夹格式为：./datasets/PCB/{class}_Img   class为变量,是下方classes列表中的元素
# 源标注文件夹格式为：./datasets/PCB/{class}_txt   class为变量,是下方classes列表中的元素
input_folder = './datasets/PCB'

# 文件夹路径, 这是目标文件夹, 格式如下
# - VOC2007
#   - Annotations
#   - ImageSets
#     - Main
#       - test.txt
#       - train.txt
#       - trainval.txt
#       - val.txt
#   - JPEGImages
output_folder = './datasets/VOC2007'

# 类别对应关系
classes = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

def create_voc_xml(image_name, width, height, objects):
    # 创建根节点
    annotation = ET.Element("annotation")

    # 创建子节点并设置其值
    folder = ET.SubElement(annotation, "folder")
    folder.text = f"{output_folder}/JPEGImages"

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_name

    size = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size, "depth")
    depth_elem.text = "3"  # 假设所有图像都是RGB

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj_elem, "name")
        name.text = classes[obj['class_num']]
        pose = ET.SubElement(obj_elem, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj_elem, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj_elem, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj_elem, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(obj['xmin'])
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(obj['ymin'])
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(obj['xmax'])
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(obj['ymax'])

    # 格式化XML
    xml_str = ET.tostring(annotation)
    dom = parseString(xml_str)
    return dom.toprettyxml(indent="    ")

def convert_to_voc():
    test_list = [] # 用于存储测试集的文件名 占0.10
    train_list = [] # 用于存储训练集的文件名 占0.7
    trainval_list = [] # 用于存储训练验证集的文件名 占0.1
    val_list = [] # 用于存储验证集的文件名 占0.1
    for PCB_class in classes:
        txt_folder = f"{input_folder}/{PCB_class}_txt"
        img_folder = f"{input_folder}/{PCB_class}_Img"
        for img in os.listdir(img_folder):
            if img.endswith('.bmp'):
                # 读取图片
                image_path = os.path.join(img_folder, img)
                im = Image.open(image_path)
                img_width, img_height = im.size
                image_name = img
                with open(os.path.join(txt_folder, img.replace('.bmp', '.txt')), 'r') as f:
                    lines = f.readlines()

                objects = []
                for line in lines:
                    parts = line.strip().split()
                    class_num = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # 假设图像尺寸，实际情况应该读取图像文件以获取尺寸

                    xmin = int((x_center - w / 2) * img_width)
                    ymin = int((y_center - h / 2) * img_height)
                    xmax = int((x_center + w / 2) * img_width)
                    ymax = int((y_center + h / 2) * img_height)

                    # 在im 中画出标注框

                    objects.append({
                        'class_num': class_num,
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    })

                voc_xml = create_voc_xml(image_name, img_width, img_height, objects)
                with open(os.path.join(output_folder + '/Annotations', img.replace('.bmp', '.xml')), 'w') as f:
                    f.write(voc_xml)
                # 将图像复制到VOC2007/JPEGImages文件夹
                im.save(os.path.join(output_folder + '/JPEGImages', img.replace('.bmp', '.jpg')))
                # 以上述概率确定其在哪个集合内
                # 生成一个小于10,大于等于0的随机数
                r = random.randint(0, 9)
                if r == 0:
                    test_list.append(img.replace('.bmp', ''))
                elif r == 1:
                    trainval_list.append(img.replace('.bmp', ''))
                elif r == 2:
                    val_list.append(img.replace('.bmp', ''))
                else:
                    train_list.append(img.replace('.bmp', ''))
    # 保存
    with open(os.path.join(output_folder, 'ImageSets/Main/test.txt'), 'w') as f:
        for img in test_list:
            f.write(img + '\n')
    with open(os.path.join(output_folder, 'ImageSets/Main/train.txt'), 'w') as f:
        for img in train_list:
            f.write(img + '\n')
    with open(os.path.join(output_folder, 'ImageSets/Main/trainval.txt'), 'w') as f:
        for img in trainval_list:
            f.write(img + '\n')
    with open(os.path.join(output_folder, 'ImageSets/Main/val.txt'), 'w') as f:
        for img in val_list:
            f.write(img + '\n')

def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image

    return M




def convert_source_to_voc():
    input_folder = './datasets/PCB_DATASET'
    class_list = ["Missing_hole","Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
    output_folder = './datasets/VOC2007'
    train_list = []
    test_list = []
    val_list = []
    trainval_list = []
    os.makedirs(output_folder, exist_ok=True)
    for PCB_class in class_list:
        annotation_folder = os.path.join(f"{input_folder}/Annotations", PCB_class)
        img_folder = os.path.join(f"{input_folder}/images", PCB_class)
        rotation_folder = os.path.join(f"{input_folder}/rotation", f"{PCB_class}_rotation")
        rotation_text_file = os.path.join(f"{input_folder}/rotation", f"{PCB_class}_angles.txt")
        os.makedirs(os.path.join(output_folder, 'Annotations'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'JPEGImages'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'ImageSets/Main'), exist_ok=True)
        rotation_dict = {}
        with open(rotation_text_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            rotation_dict[parts[0]] = int(parts[1])
        annotation_list = os.listdir(annotation_folder)
        for annotation in annotation_list:
            if annotation.endswith('.xml'):
                # 读取图片
                annotation_path = os.path.join(annotation_folder, annotation)
                image_path = os.path.join(img_folder, annotation.replace('.xml', '.jpg'))
                im = cv2.imread(image_path)
                rotation_path = os.path.join(rotation_folder, annotation.replace('.xml', '.jpg'))
                rotation_im = cv2.imread(rotation_path)
                rotation_angle = rotation_dict[annotation.replace('.xml', '')]

                M = rotate_bound_white_bg(im, rotation_angle)
                # 读取xml文件
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                # 更改folder
                folder = root.find('folder')
                folder.text = 'JPEGImages'
                # 更改path
                path = root.find('path')
                path.text = os.path.join(output_folder + '/JPEGImages', annotation.replace('.xml', '.jpg'))
                tree.write(os.path.join(output_folder + '/Annotations', annotation))
                # 更改object
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    # 旋转坐标
                    pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype='float32')
                    # 旋转后的坐标
                    rotated_pts = cv2.transform(np.array([pts]), M)[0]
                    # 旋转后的坐标
                    xmin = int(min(rotated_pts[:, 0]))
                    ymin = int(min(rotated_pts[:, 1]))
                    xmax = int(max(rotated_pts[:, 0]))
                    ymax = int(max(rotated_pts[:, 1]))
                    bndbox.find('xmin').text = str(xmin)
                    bndbox.find('ymin').text = str(ymin)
                    bndbox.find('xmax').text = str(xmax)
                    bndbox.find('ymax').text = str(ymax)
                # 更改文件名
                root.find('filename').text = annotation.replace('.xml', f'_{str(rotation_angle)}.jpg')
                # 更改path
                path = root.find('path')
                path.text = os.path.join(output_folder + '/JPEGImages', annotation.replace('.xml', f'_{str(rotation_angle)}.jpg'))
                # 更改shape
                size = root.find('size')
                size.find('width').text = str(rotation_im.shape[1])
                size.find('height').text = str(rotation_im.shape[0])
                # 保存xml文件
                tree.write(os.path.join(output_folder + '/Annotations', annotation.replace('.xml', f'_{str(rotation_angle)}.xml')))
                # 复制图片到VOC2007/JPEGImages
                shutil.copy(image_path, os.path.join(output_folder + '/JPEGImages', annotation.replace('.xml', '.jpg')))
                # 复制旋转后的图片到VOC2007/JPEGImages
                shutil.copy(rotation_path, os.path.join(output_folder + '/JPEGImages', annotation.replace('.xml', f'_{str(rotation_angle)}.jpg')))
                # 以上述概率确定其在哪个集合内
                # 生成一个小于10,大于等于0的随机数
                r = random.randint(0, 9)
                if r == 0:
                    test_list.append(annotation.replace('.xml', ''))
                    test_list.append(annotation.replace('.xml', f'_{str(rotation_angle)}'))
                elif r == 1:
                    trainval_list.append(annotation.replace('.xml', ''))
                    trainval_list.append(annotation.replace('.xml', f'_{str(rotation_angle)}'))
                elif r == 2:
                    val_list.append(annotation.replace('.xml', ''))
                    val_list.append(annotation.replace('.xml', f'_{str(rotation_angle)}'))
                else:
                    train_list.append(annotation.replace('.xml', ''))
                    train_list.append(annotation.replace('.xml', f'_{str(rotation_angle)}'))
    # 保存
    with open(os.path.join(output_folder, 'ImageSets/Main/test.txt'), 'w') as f:
        for img in test_list:
            f.write(img + '\n')
    with open(os.path.join(output_folder, 'ImageSets/Main/train.txt'), 'w') as f:
        for img in train_list:
            f.write(img + '\n')
    with open(os.path.join(output_folder, 'ImageSets/Main/trainval.txt'), 'w') as f:
        for img in trainval_list:
            f.write(img + '\n')
    with open(os.path.join(output_folder, 'ImageSets/Main/val.txt'), 'w') as f:
        for img in val_list:
            f.write(img + '\n')


def draw_bounding_box():
    input_folder = './datasets/VOC2007'
    output_folder = './datasets/visual'
    os.makedirs(output_folder, exist_ok=True)
    annotation_folder = os.path.join(input_folder, 'Annotations')
    img_folder = os.path.join(input_folder, 'JPEGImages')
    annotation_list = os.listdir(annotation_folder)
    for annotation in annotation_list:
        if annotation.endswith('.xml'):
            # 读取图片
            annotation_path = os.path.join(annotation_folder, annotation)
            image_path = os.path.join(img_folder, annotation.replace('.xml', '.jpg'))
            im = cv2.imread(image_path)
            # 读取xml文件
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_folder, annotation.replace('.xml', '.jpg')), im)
    
# from detectron2.data.detection_utils import read_image 这是read_image的导入方式
# def read_image(file_name, format=None):
#     """
#     Read an image into the given format.
#     Will apply rotation and flipping if the image has such exif information.

#     Args:
#         file_name (str): image file path
#         format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

#     Returns:
#         image (np.ndarray):
#             an HWC image in the given format, which is 0-255, uint8 for
#             supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
#     """
#     with PathManager.open(file_name, "rb") as f:
#         image = Image.open(f)

#         # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
#         image = _apply_exif_orientation(image)
#         return convert_PIL_to_numpy(image, format)

# image 应该由上述函数返回,format为"BGR"(提供一个更改函数输入参数为path的方法)
def get_JSON_result(image):
    # 设置并行处理(多进程)
    mp.set_start_method("spawn", force=True)
    # 设置config文件
    cfg = setup_cfg()
    # 创建demo模型
    demo = VisualizationDemo(cfg)
    # 运行模型,得到结果, 第二个参数是可视化结果(一张图片,即我发在群里的那张图片)
    predictions, _ = demo.run_on_image(image)
    data = {}
    CLASS_NAMES = ('missing_hole','mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper')
    # 转化为json格式
    data = {
        "detection_classes": [CLASS_NAMES[i] for i in predictions["instances"].pred_classes.tolist()],
        "detection_boxes": [
            # [x1, y1, x2, y2] for x1, y1, x2, y2 in predictions["instances"].pred_boxes.tensor.tolist()
            [int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in predictions["instances"].pred_boxes.tensor.tolist()
        ],
        "detection_scores": [
            score for score in predictions["instances"].scores.tolist()
        ]
    }
    # 返回json格式的字符串 如需返回字典,则return data
    return json.dumps(data, indent=4)

from detectron2.data.detection_utils import read_image

if __name__ == "__main__":
    # -datasets
    #   - VOC2007
    #       - Annotations
    #       - ImageSets
    #           - Main
    #               - test.txt
    #               - train.txt
    #               - trainval.txt
    #               - val.txt
    #       - JPEGImages
    # convert_source_to_voc()
    # draw_bounding_box()
    
    # 返回的score是置信度,越高越可信, 但是每次都不一样, 可能和扩散模型的随机性有关
    print(get_JSON_result(read_image('./datasets/VOC2007/JPEGImages/01_missing_hole_01_4.jpg', format="BGR")))
