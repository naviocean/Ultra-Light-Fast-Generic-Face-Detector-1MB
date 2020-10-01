from onnxruntime.quantization import quantize, QuantizationMode
import onnx
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
import sys
from onnxruntime_tools import optimizer
import torch.onnx

from vision.ssd.config.fd_config import define_img_size


def print_size_of_model(model):
    import os
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


if __name__ == "__main__":
    input_img_size = 320
    define_img_size(input_img_size)

    net_type = "slim"  # inference faster,lower precision
    # net_type = "RFB"  # inference lower,higher precision

    # label_path = "models/voc-model-labels.txt"
    label_path = "./models/train-version-slim/voc-model-labels.txt"

    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)

    if net_type == 'slim':
        model_path = "models/train-version-slim-new/slim-Epoch-0-Loss-5.180820137023926.pth"
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
