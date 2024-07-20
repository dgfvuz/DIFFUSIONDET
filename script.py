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



    

if __name__ == "__main__":
    convert_to_voc()
