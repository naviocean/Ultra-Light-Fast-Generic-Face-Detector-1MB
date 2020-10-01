"""
This code is used to convert the pytorch model into an onnx format model.
"""
from onnxruntime.quantization import quantize, QuantizationMode
import onnx
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
import sys
# from onnxruntime_tools import optimizer
import torch.onnx
import time


from vision.ssd.config.fd_config import define_img_size


def print_size_of_model(model):
    import os
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def fuse_model():
    input_img_size = 320
    define_img_size(input_img_size)
    net_type = "slim"  # inference faster,lower precision
    label_path = "./models/train-version-slim-480/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)
    if net_type == 'slim':
        model_path = "models/train-version-slim/slim-Epoch-95-Loss-2.1851983154029178.pth"
        # model_path = "models/pretrained/version-slim-320.pth"
        net = create_mb_tiny_fd(len(class_names), is_test=True, device='cpu')
    elif net_type == 'RFB':
        model_path = "models/pretrained/version-RFB-320.pth"
        # model_path = "models/pretrained/version-RFB-640.pth"
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True)
    else:
        print("unsupport network type.")
        sys.exit(1)
    # net.load(model_path)
    net.eval()
    net.to("cpu")
    print_size_of_model(net)
    dummy_input = torch.randn(1, 3, 240, 320).to("cpu")
    start = time.time()
    for i in range(0, 20):
        net(dummy_input)
    fps = 20/(time.time()-start)
    print("FPS-320 {}".format(fps))

    dummy_input = torch.randn(1, 3, 480, 640).to("cpu")
    start = time.time()
    for i in range(0, 20):
        net(dummy_input)
    fps = 20/(time.time()-start)
    print("FPS-640 {}".format(fps))

    net.fuse()
    print_size_of_model(net)
    dummy_input = torch.randn(1, 3, 240, 320).to("cpu")
    start = time.time()
    for i in range(0, 20):
        net(dummy_input)
    fps = 20/(time.time()-start)
    print("FPS-320 {}".format(fps))

    dummy_input = torch.randn(1, 3, 480, 640).to("cpu")
    start = time.time()
    for i in range(0, 20):
        net(dummy_input)
    fps = 20/(time.time()-start)
    print("FPS-640 {}".format(fps))

    # net.info(True)
    # print(net)
    # net.fuse()
    # print(net)


if __name__ == '__main__':
    fuse_model()
