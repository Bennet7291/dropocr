//! SVTR 文字识别推理会话。
//!
//! 与检测模块用相同的加载方式：从内嵌的模型字节数据
//! （`include_bytes!`）通过 `model_for_read` 直接读入内存，不落盘。

use ndarray::Array3;
use std::io::Cursor;
use tract_onnx::prelude::*;

pub struct RecInferenceSession {
    plan: TypedRunnableModel<TypedModel>,
}

#[derive(Debug)]
pub enum RecInferenceError {
    ModelLoad(String),
    Inference(String),
    UnexpectedOutputShape,
}

impl std::fmt::Display for RecInferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecInferenceError::ModelLoad(msg) => write!(f, "识别模型加载失败: {msg}"),
            RecInferenceError::Inference(msg) => write!(f, "识别推理失败: {msg}"),
            RecInferenceError::UnexpectedOutputShape => write!(f, "识别模型输出形状异常"),
        }
    }
}
impl std::error::Error for RecInferenceError {}

impl RecInferenceSession {
    pub fn load(model_bytes: &[u8]) -> Result<Self, RecInferenceError> {
        let mut cursor = Cursor::new(model_bytes);
        let mut inference_model = tract_onnx::onnx()
            .with_ignore_output_shapes(true)
            .model_for_read(&mut cursor)
            .map_err(|e| RecInferenceError::ModelLoad(e.to_string()))?;

        // 识别模型输入固定高度 48（preprocessing::REC_TARGET_HEIGHT），
        // 宽度和 batch 维度用符号维度，交给 tract 在推理时按实际输入形状处理。
        let batch = inference_model.symbols.sym("batch");
        let width = inference_model.symbols.sym("width");
        inference_model
            .set_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![
                        batch.into(),
                        TDim::from(3),
                        TDim::from(crate::ocr::preprocessing::REC_TARGET_HEIGHT as i64),
                        width.into()
                    ],
                ),
            )
            .map_err(|e| RecInferenceError::ModelLoad(e.to_string()))?;

        let plan = inference_model
            .into_optimized()
            .and_then(|m| m.into_runnable())
            .map_err(|e| RecInferenceError::ModelLoad(e.to_string()))?;

        Ok(Self { plan })
    }

    /// 输入形状 [batch, 3, 48, width]，输出形状 [batch, time_steps, num_classes]。
    pub fn run(&self, input: ndarray::Array4<f32>) -> Result<Array3<f32>, RecInferenceError> {
        let outputs = self
            .plan
            .run(tvec!(input.into_dyn().into_tvalue()))
            .map_err(|e| RecInferenceError::Inference(e.to_string()))?;

        let output_tensor = outputs
            .into_iter()
            .next()
            .ok_or(RecInferenceError::UnexpectedOutputShape)?;

        let view = output_tensor
            .to_array_view::<f32>()
            .map_err(|e| RecInferenceError::Inference(e.to_string()))?;
        let view3 = view
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| RecInferenceError::UnexpectedOutputShape)?;

        Ok(view3.to_owned())
    }
}

/// 把每个样本裁剪前的"有效宽度"换算成识别模型输出序列里的"有效时间步数"。
///
/// SVTR/CRNN 类模型对宽度做固定倍率下采样，具体倍率随模型结构而定，
/// 这里不硬编码倍率，而是用 `time_steps / max_width` 反推实际下采样比例，
/// 与 batch 内所有样本共享同一个 padding 后的 `max_width`，
/// 这样对任意下采样倍率的识别模型都适用，不需要针对具体模型改代码。
///
/// 这个换算避免了 padding 区域（图片右侧补的空白）被强行解码成噪声字符。
pub fn compute_valid_timesteps(
    valid_widths: &[u32],
    max_width: u32,
    time_steps: usize,
) -> Vec<usize> {
    let scale = if max_width > 0 {
        time_steps as f32 / max_width as f32
    } else {
        0.0
    };

    valid_widths
        .iter()
        .map(|&width| {
            let mut steps = if scale > 0.0 {
                (scale * width as f32).round() as isize
            } else {
                time_steps as isize
            };
            if steps < 1 {
                steps = 1;
            }
            if steps as usize > time_steps {
                steps = time_steps as isize;
            }
            steps as usize
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_timesteps_scale_proportionally() {
        // max_width=320 对应 time_steps=80 时，下采样比例是 4:1
        let steps = compute_valid_timesteps(&[160, 320, 40], 320, 80);
        assert_eq!(steps, vec![40, 80, 10]);
    }

    #[test]
    fn valid_timesteps_never_exceed_total() {
        let steps = compute_valid_timesteps(&[1000], 320, 80);
        assert_eq!(steps, vec![80]);
    }

    #[test]
    fn valid_timesteps_at_least_one() {
        let steps = compute_valid_timesteps(&[0], 320, 80);
        assert_eq!(steps, vec![1]);
    }
}
