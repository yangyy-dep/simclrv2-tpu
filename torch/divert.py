import torch
import torch.onnx
from net import SimCLRStage2
model = SimCLRStage2(num_class=10)
model.eval()
checkpoint = torch.load("simclrv2.pth", map_location="cpu")
model.load_state_dict(checkpoint)
input = torch.randn(1, 3, 32, 32, requires_grad=True)
torch.onnx.export(model,
        input,
        'simclrv2_bm1684x.onnx', # name of the exported onnx model
        opset_version=15,
        export_params=True,
        do_constant_folding=True)