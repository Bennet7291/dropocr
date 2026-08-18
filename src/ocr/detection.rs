//! DBNet 文字检测推理会话。
//!
//! 关键点：模型不从磁盘路径加载，而是通过 `tract_onnx::onnx().model_for_read()`
//! 从内存中的字节切片（由 `include_bytes!` 嵌入进最终二进制）直接读取，
//! 运行期不产生任何模型相关的磁盘文件。

use ndarray::Array2;
use std::io::Cursor;
use tract_onnx::prelude::*;

pub struct DetInferenceSession {
    plan: TypedRunnableModel<TypedModel>,
}

#[derive(Debug)]
pub enum DetInferenceError {
    ModelLoad(String),
    Inference(String),
    UnexpectedOutputShape,
}

impl std::fmt::Display for DetInferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetInferenceError::ModelLoad(msg) => write!(f, "检测模型加载失败: {msg}"),
            DetInferenceError::Inference(msg) => write!(f, "检测推理失败: {msg}"),
            DetInferenceError::UnexpectedOutputShape => write!(f, "检测模型输出形状异常"),
        }
    }
}
impl std::error::Error for DetInferenceError {}

impl DetInferenceSession {
    /// 从内嵌的 ONNX 模型字节数据构造推理会话。
    ///
    /// 输入尺寸在这里就固定下来（而不是像 pure-onnx-ocr 原实现那样
    /// 按每次推理的图片尺寸动态重建计算图）。这是为了让加载逻辑更简单：
    /// 拖拽单图工具场景下，每次进程运行只处理有限几张图，
    /// 用动态符号维度 + tract 的形状推断即可覆盖，不需要按尺寸缓存多份计划。
    pub fn load(model_bytes: &[u8]) -> Result<Self, DetInferenceError> {
        let mut cursor = Cursor::new(model_bytes);
        let mut inference_model = tract_onnx::onnx()
            .with_ignore_output_shapes(true)
            .model_for_read(&mut cursor)
            .map_err(|e| DetInferenceError::ModelLoad(e.to_string()))?;

        let height = inference_model.symbols.sym("height");
        let width = inference_model.symbols.sym("width");
        inference_model
            .set_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![TDim::from(1), TDim::from(3), height.into(), width.into()],
                ),
            )
            .map_err(|e| DetInferenceError::ModelLoad(e.to_string()))?;

        // `InferenceModelExt::into_optimized` 一步完成：分析形状 -> 转 TypedModel
        // -> decluttering -> 算子优化，内部已含 into_typed + into_decluttered 的效果。
        let plan = inference_model
            .into_optimized()
            .and_then(|m| m.into_runnable())
            .map_err(|e| DetInferenceError::ModelLoad(e.to_string()))?;

        Ok(Self { plan })
    }

    /// 对预处理后的输入张量做一次推理，返回概率图。
    pub fn run(&self, input: ndarray::Array4<f32>) -> Result<Array2<f32>, DetInferenceError> {
        let outputs = self
            .plan
            .run(tvec!(input.into_dyn().into_tvalue()))
            .map_err(|e| DetInferenceError::Inference(e.to_string()))?;

        let output_tensor = outputs
            .into_iter()
            .next()
            .ok_or(DetInferenceError::UnexpectedOutputShape)?;

        let view = output_tensor
            .to_array_view::<f32>()
            .map_err(|e| DetInferenceError::Inference(e.to_string()))?;
        let view4 = view
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|_| DetInferenceError::UnexpectedOutputShape)?;

        // `index_axis` 在对应维度长度为 0 时会 panic（本程序设了
        // panic = "abort"，一旦触发就是整个进程直接终止，用户连一个
        // 错误说明 txt 都拿不到）。正常的 DBNet 导出模型输出形状固定是
        // [batch=1, channel>=1, H, W]，理论上不会走到这个分支，但这里
        // 是唯一一处直接依赖"真实 PaddleOCR 模型"输出形状的代码
        // （本工程从未用真实模型验证过，见 README），所以显式做一次
        // 边界检查，把任何形状异常都转成 Err 而不是让它有 panic 的机会。
        let shape = view4.shape();
        if shape[0] == 0 || shape[1] == 0 {
            return Err(DetInferenceError::UnexpectedOutputShape);
        }

        let probability_map = view4
            .index_axis(ndarray::Axis(0), 0)
            .index_axis(ndarray::Axis(0), 0)
            .to_owned();

        Ok(probability_map)
    }
}
