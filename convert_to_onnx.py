"""
This code is used to convert the pytorch model into an onnx format model.
"""
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
import sys
from onnxruntime.quantization import quantize, QuantizationMode
import torch.onnx
import onnx
from onnxruntime_tools import optimizer
from vision.ssd.config.fd_config import define_img_size

# define input size ,default optional(128/160/320/480/640/1280)
input_img_size = 320
define_img_size(input_img_size)

net_type = "slim"  # inference faster,lower precision
# net_type = "RFB"  # inference lower,higher precision

# label_path = "models/voc-model-labels.txt"
label_path = "./models/train-version-slim/voc-model-labels.txt"

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

if net_type == 'slim':
    model_path = "models/train-version-slim-new/slim_new-Epoch-150-Loss-2.590208204874552.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device='cpu')
elif net_type == 'RFB':
    model_path = "models/train-version-RFB/RFB-Epoch-95-Loss-2.0881652910458413.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device='cpu')

else:
    print("unsupport network type.")
    sys.exit(1)
net.load(model_path)
net.eval()
net.to("cpu")
if net_type == 'slim_new':
    net.fuse()
model_name = model_path.split("/")[-1].split(".")[0]
model_path = f"models/onnx/{model_name}.onnx"

dummy_input = torch.randn(1, 3, 240, 320).to("cpu")
# dummy_input = torch.randn(1, 3, 480, 640).to("cuda") #if input size is 640*480
torch.onnx.export(net, dummy_input, model_path, verbose=False, input_names=[
                  'input'], output_names=['scores', 'boxes'], opset_version=11)

model = onnx.load(model_path)

optimized_model = optimizer.optimize_model(model_path,
                                           model_type='bert',
                                           num_heads=12,
                                           hidden_size=768)
optimized_onnx_model_path = f"models/onnx/{model_name}_optimized.onnx"
optimized_model.save_model_to_file(optimized_onnx_model_path)
print('Optimized model saved at :', optimized_onnx_model_path)
print('>> quantizing..')
model = onnx.load(model_path)
quantized_model = quantize(model=model, quantization_mode=QuantizationMode.IntegerOps,
                           force_fusions=True, symmetric_weight=True)
optimized_quantized_onnx_model_path = f"models/onnx/{model_name}_ONNXquantized.onnx"
onnx.save_model(quantized_model, optimized_quantized_onnx_model_path)
print('Quantized&optimized model saved at :',
      optimized_quantized_onnx_model_path)
