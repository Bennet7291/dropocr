//! CTC 贪心解码：把 SVTR 识别模型逐时间步输出的 logits 转成文本。
//!
//! 算法逻辑：
//! 1. 每个时间步取概率/logit 最大的类别索引（argmax）
//! 2. 跳过 blank（索引 0）
//! 3. 跳过与前一个非 blank 符号相同的重复符号（标准 CTC 去重规则）
//! 4. 兼容两种模型输出形态：已经过 softmax 的概率分布，或原始 logits
//!    （原始 logits 时用 softmax 现算置信度）

use crate::ocr::dictionary::RecDictionary;
use ndarray::{s, Array3, Axis};

/// 单条序列的解码结果。
#[derive(Debug, Clone)]
pub struct DecodedSequence {
    pub text: String,
    pub confidence: f32,
}

/// 对一个 batch 的识别 logits 做贪心 CTC 解码。
///
/// - `logits`: 形状 `[batch, time_steps, num_classes]`
/// - `valid_timesteps`: 每条序列实际有效的时间步数（用于处理 padding）
/// - `dictionary`: 索引到字符的映射表
pub fn ctc_greedy_decode(
    logits: &Array3<f32>,
    valid_timesteps: &[usize],
    dictionary: &RecDictionary,
) -> Vec<DecodedSequence> {
    let batch_size = logits.len_of(Axis(0));
    let time_steps = logits.len_of(Axis(1));
    let blank_id = dictionary.blank_id();

    let mut results = Vec::with_capacity(batch_size);

    for batch_index in 0..batch_size {
        let max_steps = valid_timesteps
            .get(batch_index)
            .copied()
            .unwrap_or(time_steps)
            .min(time_steps);

        let mut previous_symbol: Option<usize> = None;
        let mut text = String::new();
        let mut confidence_sum = 0.0f64;
        let mut confidence_count = 0usize;

        for t in 0..max_steps {
            let step = logits.slice(s![batch_index, t, ..]);

            let mut best_index = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            let mut row_sum = 0.0f32;
            let mut min_value = f32::INFINITY;
            let mut max_value = f32::NEG_INFINITY;

            for (idx, &value) in step.iter().enumerate() {
                if value > best_value {
                    best_value = value;
                    best_index = idx;
                }
                row_sum += value;
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            }

            // 判断这一步输出是否已经是概率分布（PaddleOCR 导出的 ONNX
            // 通常在模型内部已经做了 softmax），否则退化为对 logits 现算 softmax。
            let looks_like_probability = min_value.is_finite()
                && max_value.is_finite()
                && min_value >= -1e-4
                && max_value <= 1.0 + 1e-4
                && (row_sum - 1.0).abs() <= 1e-3;

            if best_index == blank_id {
                previous_symbol = None;
                continue;
            }
            if Some(best_index) == previous_symbol {
                continue;
            }
            previous_symbol = Some(best_index);

            let probability = if looks_like_probability {
                best_value.clamp(0.0, 1.0)
            } else {
                let max_logit = best_value as f64;
                let mut sum_exp = 0.0f64;
                for &value in step.iter() {
                    sum_exp += ((value as f64) - max_logit).exp();
                }
                if sum_exp > 0.0 {
                    (1.0 / sum_exp) as f32
                } else {
                    0.0
                }
            };

            if let Some(token) = dictionary.token(best_index) {
                text.push_str(token);
                confidence_sum += probability as f64;
                confidence_count += 1;
            }
            // 若索引超出字典范围（异常情况），静默跳过该字符，
            // 不中断整体识别流程。
        }

        let confidence = if confidence_count > 0 {
            (confidence_sum / confidence_count as f64) as f32
        } else {
            0.0
        };

        results.push(DecodedSequence { text, confidence });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict_from_tokens(tokens: &[&str]) -> RecDictionary {
        RecDictionary::from_embedded_str(&tokens.join("\n")).unwrap()
    }

    #[test]
    fn decodes_simple_sequence() {
        // 字典: blank(0), a(1), b(2)
        let dictionary = dict_from_tokens(&["a", "b"]);
        // 序列: a a b blank -> 去重去blank后应为 "ab"
        let logits = Array3::from_shape_vec(
            (1, 4, 3),
            vec![
                0.1, 5.0, 0.1, // a
                0.1, 4.5, 0.2, // a (重复，应被跳过)
                0.1, 0.1, 4.8, // b
                5.0, 0.1, 0.1, // blank
            ],
        )
        .unwrap();

        let result = ctc_greedy_decode(&logits, &[4], &dictionary);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "ab");
    }

    #[test]
    fn blank_between_repeats_allows_repeat_char() {
        let dictionary = dict_from_tokens(&["a"]);
        // a blank a -> 中间有 blank 分隔，两个 a 都应保留
        let logits = Array3::from_shape_vec(
            (1, 3, 2),
            vec![
                0.1, 5.0, // a
                5.0, 0.1, // blank
                0.1, 5.0, // a
            ],
        )
        .unwrap();

        let result = ctc_greedy_decode(&logits, &[3], &dictionary);
        assert_eq!(result[0].text, "aa");
    }
}
