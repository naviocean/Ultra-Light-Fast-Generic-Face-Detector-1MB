"""
This code is used to batch detect images in a folder.
"""
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
import argparse
import os
import sys
import time
import cv2

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser(
    description='detect_imgs')

parser.add_argument('--net_type', default="slim", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=320, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.5, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1500, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cpu", type=str,
                    help='cuda:0 or cpu')
args = parser.parse_args()
# must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
define_img_size(args.input_size)


result_path = "./detect_imgs_results"
label_path = "./models/train-version-slim/voc-model-labels.txt"
test_device = args.test_device

class_names = [name.strip() for name in open(label_path).readlines()]
if args.net_type == 'slim':
    model_path = "models/train-version-slim-new/slim_new-Epoch-150-Loss-2.590208204874552.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(
        net, model_path, candidate_size=args.candidate_size, device=test_device, fuse=True)
elif args.net_type == 'RFB':
    model_path = "models/pretrained/version-RFB-320.pth"
    # model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(
        len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(
        net, model_path, candidate_size=args.candidate_size, device=test_device, fuse=True)
else:
    print("The net type is wrong!")
    sys.exit(1)

if not os.path.exists(result_path):
    os.makedirs(result_path)
listdir = os.listdir(args.path)
sum = 0
for file_path in listdir:
    start = time.time()
    img_path = os.path.join(args.path, file_path)
    orig_image = cv2.imread(img_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(
        image, args.candidate_size / 2, args.threshold)
    sum += boxes.size(0)
    classes = ['background', 'face', 'face_mask']
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        score = f"{probs[i]:.2f}"
        label = classes[labels[i]]
        if int(labels[i]) == 1:
            cv2.rectangle(orig_image, (box[0], box[1]),
                          (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(orig_image, f"{label}-{score}",
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            print('2')
            cv2.rectangle(orig_image, (box[0], box[1]),
                          (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(orig_image, f"{label}-{score}",
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    fps = 1/(time.time() - start)
    print("FPS: {}".format(fps))
    cv2.putText(orig_image, str(boxes.size(0)), (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(result_path, file_path), orig_image)
    print(f"Found {len(probs)} faces. The output image is {result_path}")
print(sum)
