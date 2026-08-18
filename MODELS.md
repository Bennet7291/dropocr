# 模型获取与转换指南

本工程运行需要三个文件放在 `assets/` 目录下，编译时会通过
`include_bytes!`/`include_str!` 直接打包进最终 exe（运行期不依赖
这三个文件本身，只是编译期需要它们存在）：

```
assets/det.onnx       检测模型（DBNet 系列）
assets/rec.onnx       识别模型（SVTR/CRNN 系列，中英文合一）
assets/rec_dict.txt   识别字典（每行一个字符，UTF-8 编码）
```

这一步**必须在你自己的机器上完成**——本工程所在的开发沙盒无法
访问 PaddlePaddle 官方模型托管服务，也无法安装 PaddlePaddle 框架
本身来做格式转换，所以这部分工作没法替你提前做完。

以下是具体步骤。

## 第一步：安装转换所需的 Python 环境

```bash
pip install paddlepaddle paddle2onnx
# 如果只是做格式转换，不需要装 GPU 版本，CPU 版就够用
```

## 第二步：下载 PP-OCRv4 mobile 模型（推荐）

PaddleOCR 官方模型库地址（需要科学上网访问，或者从国内镜像站获取）：
https://github.com/PaddlePaddle/PaddleOCR

推荐用 **mobile 版**而不是 server 版——mobile 版本身就是为端侧部署
优化过的轻量模型，这是我们能把最终体积控制在 20MB 左右的关键前提。

需要下载两个模型：
- **中英文检测模型**：`ch_PP-OCRv4_det_infer` 系列（约 4-5MB，Paddle 原生格式）
- **中英文识别模型**：`ch_PP-OCRv4_rec_infer` 系列（约 8-10MB，Paddle 原生格式，
  这个模型本身就是中英文数字混合识别，不需要分开两个模型）

下载后会得到 Paddle 原生格式的文件夹，包含类似这些文件：
```
inference.pdmodel
inference.pdiparams
inference.pdiparams.info
```

## 第三步：转换成 ONNX 格式

```bash
# 转换检测模型
paddle2onnx \
    --model_dir ./ch_PP-OCRv4_det_infer \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file det.onnx \
    --opset_version 13 \
    --enable_onnx_checker True

# 转换识别模型
paddle2onnx \
    --model_dir ./ch_PP-OCRv4_rec_infer \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file rec.onnx \
    --opset_version 13 \
    --enable_onnx_checker True
```

**重要**：转换后的模型输入维度可能是固定尺寸（比如检测模型输入固定
`[1,3,960,960]`），而本工程的 `detection.rs`/`recognition.rs` 假设
的是**动态维度**输入（用 `height`/`width`/`batch` 符号维度，这样才能
处理任意尺寸的图片）。如果转换出来的 ONNX 模型输入维度是写死的数字
而不是符号，需要在转换时加上动态维度参数，或者转换后用以下脚本把
固定维度改成动态维度：

```python
import onnx

model = onnx.load("det.onnx")
# 把输入的第 2、3 维（高、宽）改成动态符号维度
input_tensor = model.graph.input[0]
input_tensor.type.tensor_type.shape.dim[2].dim_param = "height"
input_tensor.type.tensor_type.shape.dim[3].dim_param = "width"
onnx.save(model, "det.onnx")
```

识别模型同理，把 batch 维和宽度维改成动态符号（高度维固定为 48，
和本工程 `preprocessing.rs` 里的 `REC_TARGET_HEIGHT` 常量保持一致）：

```python
import onnx

model = onnx.load("rec.onnx")
input_tensor = model.graph.input[0]
input_tensor.type.tensor_type.shape.dim[0].dim_param = "batch"
input_tensor.type.tensor_type.shape.dim[3].dim_param = "width"
onnx.save(model, "rec.onnx")
```

转换完成后，建议用 `onnx.checker.check_model()` 和 `onnxruntime`
跑一次推理做基本验证，确认模型本身没问题，再拿去给 Rust 工程用。

## 第四步：（强烈建议）量化压缩模型体积

为了达到 20MB 以内的目标，建议对转换后的 ONNX 模型做 **int8 动态量化**：

```bash
pip install onnxruntime
python3 -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('det.onnx', 'det_quant.onnx', weight_type=QuantType.QInt8)
quantize_dynamic('rec.onnx', 'rec_quant.onnx', weight_type=QuantType.QInt8)
"
```

量化通常能把模型体积压缩到原来的 25%-50%，识别精度损失一般很小
（几个百分点以内）。量化后重命名/替换成 `det.onnx`、`rec.onnx`。

**需要注意**：量化模型使用的算子集合可能比原始 float32 模型更复杂
（引入了 `QuantizeLinear`/`DequantizeLinear`/`QLinearConv` 等算子），
需要确认 `tract-onnx` 对这些算子的支持程度。如果 tract 加载量化模型
报"unsupported operator"之类的错误，退回不量化的 float32 版本，
把 20MB 目标适当放宽（比如 30-40MB）。这是体积和实现复杂度之间的
真实取舍，建议先用未量化版本跑通整个流程，验证识别效果符合预期后，
再尝试量化优化体积。

## 第五步：获取识别字典

字典文件在 PaddleOCR 仓库里，路径通常是：
```
ppocr/utils/ppocr_keys_v1.txt        # 早期版本
ppocr/utils/dict/ppocr_keys_v1.txt   # 新版本路径可能变化，以仓库实际为准
```

直接下载这个文件，改名为 `rec_dict.txt`，确保是 **UTF-8 编码、
每行一个字符、不含行号或其他标注**。

本工程的 `dictionary.rs` 会自动在字典最前面插入 `blank` token
（索引固定为 0），这个逻辑已经对照 PaddleOCR 官方
`rec_postprocess.py::CTCLabelDecode.add_special_char` 源码核实过，
**不需要你在字典文件里手动加 blank**，直接用官方原始字典文件即可。

## 第六步：放置文件并编译

```
your-project/
├── Cargo.toml
├── src/
└── assets/
    ├── det.onnx
    ├── rec.onnx
    └── rec_dict.txt
```

放好之后正常 `cargo build --release` 即可，模型会被编译进最终二进制。
