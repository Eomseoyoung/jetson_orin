import torch
from bisenetv2 import BiSeNetV2  # 모델 클래스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1) 모델 로드
model = BiSeNetV2(num_classes=2)
ckpt = torch.load("bisenetv2_rail3.pth", map_location=device)
model.load_state_dict(ckpt)
model.eval().to(device)
# 2) 더미 입력
dummy = torch.randn(1, 3, 512, 1024).to(device)
# 3) ONNX로 내보내기
torch.onnx.export(
    model, dummy, "bisenetv2.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
print("✅ Saved bisenetv2.onnx")
