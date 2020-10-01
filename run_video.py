"""
This code uses the pytorch model to detect faces from live video or camera.
"""
from vision.utils.misc import Timer
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_fd_new import create_mb_tiny_fd_new, create_mb_tiny_fd_predictor_new
import argparse
import sys
import cv2
import time
from vision.ssd.config.fd_config import define_img_size
import torch
import numpy as np

parser = argparse.ArgumentParser(
    description='detect_video')

parser.add_argument('--net_type', default="slim", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=320, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.75, type=float,
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
        model_path = "models/train-version-slim-new/slim_new-Epoch-150-Loss-2.590208204874552.pth"
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
    net.eval()
    net.fuse()
    return predictor


def run_video():
    timer = Timer()
    sum = 0
    cap = cv2.VideoCapture('ne1.avi')  # capture from camera
    predictor = load_model()
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920, 1080))
    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            break
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        timer.start()
        start = time.time()
        boxes, labels, probs = predictor.predict(
            image, candidate_size / 2, threshold)
        # if labels.size(0) == 0:
        # continue
        # else:
        interval = timer.end()
        fps = 1/(time.time()-start)
        print('FPS: {} Time: {:.6f}s, Detect Objects: {:d}.'.format(
            fps, interval, labels.size(0)))
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            score = f" {probs[i]:.2f}"
            label = labels[i]
            cv2.rectangle(orig_image, (box[0], box[1]),
                          (box[2], box[3]), (0, 255, 0), 4)

            cv2.putText(orig_image, "{}-{}".format(score, class_names[label]),
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # font scale
                        (0, 0, 255),
                        2)  # line type
        orig_image = cv2.resize(orig_image, None, None, fx=0.8, fy=0.8)
        sum += boxes.size(0)
        cv2.imshow('annotated', orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # out.write(orig_image)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("all face num:{}".format(sum))


if __name__ == "__main__":
    run_video()
