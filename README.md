# jetson_orin
> ### jetson orin 경량화
> jetson orin 내부에서 pth를 1) ONNX로 변환후 2) TensorRT로 경량화 하는 방법에 대해 기술합니다.


> ### onnx to tensorRT
> onnx를 tensorRT로 변환 커맨드 명령어(jetson 내부에서 실행)



      /usr/src/tensorrt/bin/trtexec \
        --onnx=rail.onnx \
        --saveEngine=rail_fp16.plan \
        --fp16

> ### jetson 가속화 명령
>       sudo nvpmodel -m 0
>       sudo jetson_clocks
