//! OCR 核心模块：检测 + 识别完整流水线。
//!
//! 模型与字典均通过 `include_bytes!` / `include_str!` 在编译期
//! 直接打包进最终二进制，运行期不依赖任何外部文件。

pub mod ctc;
pub mod detection;
pub mod dictionary;
pub mod postprocessing;
pub mod preprocessing;
pub mod recognition;

use detection::DetInferenceSession;
use dictionary::RecDictionary;
use image::DynamicImage;
use recognition::RecInferenceSession;

/// 检测模型字节数据。
///
/// 占位提示：请将转换好的 PP-OCRv4/v5 mobile 检测模型（ONNX 格式，
/// 建议 int8 量化以控制体积）放到 `assets/det.onnx`，再执行编译。
/// 具体获取/转换步骤见仓库根目录 `MODELS.md`。
static DET_MODEL_BYTES: &[u8] = include_bytes!("../../assets/det.onnx");

/// 识别模型字节数据（中英文合一识别模型）。
static REC_MODEL_BYTES: &[u8] = include_bytes!("../../assets/rec.onnx");

/// 识别字典（每行一个字符，PaddleOCR 格式，UTF-8 编码）。
static REC_DICTIONARY_TEXT: &str = include_str!("../../assets/rec_dict.txt");

#[derive(Debug)]
pub enum OcrError {
    Dictionary(dictionary::DictionaryError),
    Detection(detection::DetInferenceError),
    Recognition(recognition::RecInferenceError),
    Preprocess(preprocessing::PreprocessError),
}

impl std::fmt::Display for OcrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OcrError::Dictionary(e) => write!(f, "{e}"),
            OcrError::Detection(e) => write!(f, "{e}"),
            OcrError::Recognition(e) => write!(f, "{e}"),
            OcrError::Preprocess(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for OcrError {}

/// 一个识别结果：文本内容 + 置信度 + 在原图中的位置。
#[derive(Debug, Clone)]
pub struct OcrLine {
    pub text: String,
    pub confidence: f32,
}

/// 完整 OCR 引擎：持有检测/识别模型和字典，可重复调用处理多张图片。
pub struct OcrEngine {
    det: DetInferenceSession,
    rec: RecInferenceSession,
    dictionary: RecDictionary,
}

impl OcrEngine {
    /// 从内嵌资源构造引擎。全程不访问文件系统。
    pub fn new() -> Result<Self, OcrError> {
        let dictionary =
            RecDictionary::from_embedded_str(REC_DICTIONARY_TEXT).map_err(OcrError::Dictionary)?;
        let det = DetInferenceSession::load(DET_MODEL_BYTES).map_err(OcrError::Detection)?;
        let rec = RecInferenceSession::load(REC_MODEL_BYTES).map_err(OcrError::Recognition)?;

        Ok(Self { det, rec, dictionary })
    }

    /// 对一张已解码的图片执行完整 OCR 流程，返回按阅读顺序排列的文本行。
    pub fn recognize(&self, image: &DynamicImage) -> Result<Vec<OcrLine>, OcrError> {
        let det_input =
            preprocessing::preprocess_detection(image, preprocessing::DET_LIMIT_SIDE_LEN)
                .map_err(OcrError::Preprocess)?;

        let probability_map = self
            .det
            .run(det_input.data)
            .map_err(OcrError::Detection)?;

        let (orig_w, orig_h) = {
            use image::GenericImageView;
            image.dimensions()
        };

        let regions = postprocessing::extract_text_regions(
            &probability_map,
            (orig_w, orig_h),
            det_input.scale_ratio,
            postprocessing::DET_THRESH,
            postprocessing::DET_UNCLIP_RATIO,
        );

        if regions.is_empty() {
            return Ok(Vec::new());
        }

        // 识别阶段按 batch 一次性跑完检测到的所有区域，减少推理调用次数。
        let rec_batch =
            preprocessing::preprocess_recognition(image, &regions).map_err(OcrError::Preprocess)?;

        if rec_batch.valid_widths.is_empty() {
            return Ok(Vec::new());
        }

        let logits = self.rec.run(rec_batch.data).map_err(OcrError::Recognition)?;
        let time_steps = logits.shape()[1];

        // 把每个区域裁剪后的实际宽度换算成模型输出序列中的有效步数，
        // 避免 padding 区域被当成正常内容解码出乱码字符。
        let valid_timesteps = recognition::compute_valid_timesteps(
            &rec_batch.valid_widths,
            rec_batch.max_width,
            time_steps,
        );

        let decoded = ctc::ctc_greedy_decode(&logits, &valid_timesteps, &self.dictionary);

        let lines = decoded
            .into_iter()
            .filter(|d| !d.text.trim().is_empty())
            .map(|d| OcrLine {
                text: d.text,
                confidence: d.confidence,
            })
            .collect();

        Ok(lines)
    }
}
