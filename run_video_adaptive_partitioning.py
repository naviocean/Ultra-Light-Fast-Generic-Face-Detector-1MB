"""
This code uses the pytorch model to detect faces from live video or camera.
"""
from vision.utils.misc import Timer
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
import argparse
import sys
import cv2
import time
from vision.ssd.config.fd_config import define_img_size
import torch
import numpy as np

parser = argparse.ArgumentParser(
    description='detect_video')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=640, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.7, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1500, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cpu", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--video_path', default="/home/linzai/Videos/video/16_1.MP4", type=str,
                    help='path of video')
args = parser.parse_args()

input_img_size = args.input_size
test_device = args.test_device
candidate_size = args.candidate_size
threshold = args.threshold
# must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
define_img_size(input_img_size)

label_path = "./models/train-version-slim/voc-model-labels.txt"
# label_path = "./models/voc-model-labels.txt"
net_type = args.net_type

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


def load_model():
    if args.net_type == 'slim':
        model_path = "models/train-version-slim-new/slim-Epoch-5-Loss-3.3751535778045656.pth"
        # model_path = "models/pretrained/version-slim-640.pth"
        net = create_mb_tiny_fd(
            len(class_names), is_test=True, device=test_device)
        predictor = create_mb_tiny_fd_predictor(
            net, candidate_size=args.candidate_size, device=test_device)
    elif args.net_type == 'RFB':
        model_path = "models/train-version-RFB-640/RFB-Epoch-95-Loss-1.9533395563301288.pth"
        # model_path = "models/pretrained/version-RFB-640.pth"
        net = create_Mb_Tiny_RFB_fd(
            len(class_names), is_test=True, device=test_device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(
            net, candidate_size=args.candidate_size, device=test_device)
    else:
        print("The net type is wrong!")
        sys.exit(1)

    net.load(model_path)
    # net.fuse_model()
    net.eval()

    return predictor


# Divide an image into multiple regions for object detection task
#  image_shape = (w,h)
#  dividers_list = A list of integer numbers. The numbers represent how many columns the row will be divided into, from top row to bottom row respectively.
#  overlap_rate = How much the regions overlap each other
def divideImage(image_shape, dividers_list, overlap_rate=0.1):
    _W = 0
    _H = 1
    rows = len(dividers_list)

    region_list = []
    baseY = 0
    for row, num_divide in enumerate(dividers_list):
        region_width = image_shape[_W]/num_divide
        overlap = region_width * overlap_rate
        for i in range(num_divide):
            x1 = i * region_width - overlap
            y1 = baseY - overlap
            x2 = (i+1) * region_width + overlap
            y2 = baseY + region_width + overlap
            if x1 < 0:
                x1 = 0
            if x1 >= image_shape[_W]:
                x1 = image_shape[_W]-1
            if y1 < 0:
                y1 = 0
            if y1 >= image_shape[_H]:
                y1 = image_shape[_H]-1
            if x2 < 0:
                x2 = 0
            if x2 >= image_shape[_W]:
                x2 = image_shape[_W]-1
            if y2 < 0:
                y2 = 0
            if y2 >= image_shape[_H]:
                y2 = image_shape[_H]-1
            region_list.append((int(x1), int(y1), int(x2), int(y2)))
        baseY += region_width
    return region_list

# Calculate IOU for non-maximum suppression


def iou(a, b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    iou_x1 = np.maximum(a[0], b[0])
    iou_y1 = np.maximum(a[1], b[1])
    iou_x2 = np.minimum(a[2], b[2])
    iou_y2 = np.minimum(a[3], b[3])

    iou_w = iou_x2 - iou_x1
    iou_h = iou_y2 - iou_y1

    if iou_w < 0 or iou_h < 0:
        return 0.0

    area_iou = iou_w * iou_h
    iou = area_iou / (area_a + area_b - area_iou)

    return iou


def draw_regions(img, region_list):
    colors = [
        (0,   0,   0),
        (255,   0,   0),
        (0,   0, 255),
        (255,   0, 255),
        (0, 255,   0),
        (255, 255,   0),
        (0, 255, 255)
    ]
    _W = 0
    _H = 1
    for i, region in enumerate(region_list):
        img = cv2.rectangle(
            img, (region[0], region[1]), (region[2], region[3]), colors[i % 7], 4)
    return img
# Prepare data for object detection task
#  - Crop input image based on the region list produced by divideImage()
#  - Create a list of task which consists of coordinate of the ROI in the input image, and the image of the ROI


def createObjectDectionTasks(img, region_list):
    task_id = 0
    task_list = []
    for region in region_list:
        ROI = img[region[1]:region[3], region[0]:region[2]]
        task_list.append([region, ROI])
    return task_list


def run_video():
    timer = Timer()
    sum = 0
    cap = cv2.VideoCapture('ne1.avi')  # capture from camera
    predictor = load_model()
    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            break
        orig_image = run_image(orig_image, predictor)
        cv2.imshow('annotated', orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("all face num:{}".format(sum))


def run_image(orig_image, predictor):
    start = time.time()

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    rows = round(orig_image.shape[0]/320)
    cols = round(orig_image.shape[1]/240)
    region_list = divideImage(
        (image.shape[1], image.shape[0]), [3, 3])
    # img_tmp = image.copy()
    # img_tmp = draw_regions(img_tmp, region_list)

    # cv2.imshow('regions', img_tmp)
    # cv2.waitKey(3 * 1000)

    task_list = createObjectDectionTasks(image, region_list)

    objects = []
    for task in task_list:
        inBlob = cv2.resize(task[1], (320, 240))
        boxes, labels, probs = predictor.predict_wo_scale(
            inBlob, args.candidate_size / 2, args.threshold)

        # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
        for i in range(boxes.size(0)):
            conf = probs[i]
            box = boxes[i, :]
            if conf > threshold:                              # Confidence > 60%
                ROI_shape = task[1].shape
                xmin = abs(int(box[0] * ROI_shape[1])) + task[0][0]
                ymin = abs(int(box[1] * ROI_shape[0])) + task[0][1]
                xmax = abs(int(box[2] * ROI_shape[1])) + task[0][0]
                ymax = abs(int(box[3] * ROI_shape[0])) + task[0][1]
                class_id = int(labels[i])
                objects.append([xmin, ymin, xmax, ymax, conf, class_id, True])

    # Do non-maximum suppression to reject the redundant objects on the overlap region
    for obj_id1, obj1 in enumerate(objects[:-2]):
        for obj_id2, obj2 in enumerate(objects[obj_id1+1:]):
            if obj1[6] == True and obj2[6] == True:
                IOU = iou(obj1[0:3+1], obj2[0:3+1])
                if IOU > 0.5:
                    if obj1[4] < obj2[4]:
                        obj1[6] = False
                    else:
                        obj2[6] = False
    orig_image = draw_regions(orig_image, region_list)
    # print(objects)
    # exit()
    # Draw detection result
    fps = 1/(time.time()-start)
    print('FPS: {}, Detect Objects: {:d}.'.format(
        fps, labels.size(0)))
    for obj in objects:
        if obj[6] == True:
            cv2.rectangle(
                orig_image, (obj[0], obj[1]), (obj[2], obj[3]), (0, 255, 0), 2)  # Found object
        else:
            pass
            # Object which is rejected by NMS
            cv2.rectangle(
                orig_image, (obj[0], obj[1]), (obj[2], obj[3]), (0, 0, 255), 1)

    return orig_image


if __name__ == "__main__":
    # orig_image = cv2.imread('imgs/25.jpg')
    # predictor = load_model()
    # run_image(orig_image, predictor)
    run_video()
