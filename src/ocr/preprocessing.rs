//! 图像预处理：把解码后的图片转换成模型需要的张量格式。
//!
//! 检测（DBNet）预处理：
//! - 长边缩放到 `limit_side_len`（默认 960，超过才缩放）
//! - 宽高向上取整到 32 的倍数（模型下采样步长要求）
//! - 归一化到 [0, 1]，转 NCHW 排布
//!
//! 识别（SVTR）预处理：
//! - 按检测框裁剪文字区域
//! - 保持宽高比缩放到固定高度 48，宽度上限 320（超宽的按最大宽度截断/等比缩放）
//! - 用 mean=0.5, std=0.5 归一化到 [-1, 1]（PaddleOCR 标准归一化）

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{Array3, Array4, Axis};

pub const DET_LIMIT_SIDE_LEN: u32 = 960;
pub const REC_TARGET_HEIGHT: u32 = 48;
pub const REC_MAX_WIDTH: u32 = 320;
const NORM_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const NORM_STD: [f32; 3] = [0.5, 0.5, 0.5];

#[derive(Debug)]
pub enum PreprocessError {
    EmptyImage,
}

impl std::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreprocessError::EmptyImage => write!(f, "输入图片宽或高为 0"),
        }
    }
}
impl std::error::Error for PreprocessError {}

/// 检测阶段预处理结果。
pub struct PreprocessedDetInput {
    /// NCHW 排布的 f32 张量数据，形状 [1, 3, padded_h, padded_w]
    pub data: Array4<f32>,
    /// 缩放+padding 之后的宽高（也是模型输入的宽高）
    pub resized_dims: (u32, u32),
    /// 原图 -> 缩放后图的比例（用于把检测框坐标映射回原图）
    pub scale_ratio: f64,
}

pub fn preprocess_detection(
    image: &DynamicImage,
    limit_side_len: u32,
) -> Result<PreprocessedDetInput, PreprocessError> {
    let (orig_w, orig_h) = image.dimensions();
    if orig_w == 0 || orig_h == 0 {
        return Err(PreprocessError::EmptyImage);
    }

    let (resized_w, resized_h, scale_ratio) =
        compute_resized_dims(orig_w, orig_h, limit_side_len);

    let resized = if resized_w == orig_w && resized_h == orig_h {
        image.clone()
    } else {
        image.resize_exact(resized_w, resized_h, FilterType::Lanczos3)
    };

    let rgb = resized.to_rgb8();
    let padded_w = round_up_to_multiple(resized_w, 32);
    let padded_h = round_up_to_multiple(resized_h, 32);

    let mut hwc = Array3::<f32>::zeros((padded_h as usize, padded_w as usize, 3));
    for y in 0..resized_h as usize {
        for x in 0..resized_w as usize {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                hwc[[y, x, c]] = pixel[c] as f32 / 255.0;
            }
        }
    }

    let chw = hwc.permuted_axes([2, 0, 1]);
    let nchw = chw.insert_axis(Axis(0)).to_owned();

    Ok(PreprocessedDetInput {
        data: nchw,
        resized_dims: (padded_w, padded_h),
        scale_ratio,
    })
}

fn compute_resized_dims(orig_w: u32, orig_h: u32, limit_side_len: u32) -> (u32, u32, f64) {
    if limit_side_len == 0 {
        return (orig_w, orig_h, 1.0);
    }
    let limit = limit_side_len as f64;
    let max_side = orig_w.max(orig_h) as f64;
    if max_side <= limit {
        return (orig_w, orig_h, 1.0);
    }
    let scale_ratio = limit / max_side;
    let resized_w = ((orig_w as f64 * scale_ratio).round().max(1.0)) as u32;
    let resized_h = ((orig_h as f64 * scale_ratio).round().max(1.0)) as u32;
    (resized_w, resized_h, scale_ratio)
}

fn round_up_to_multiple(value: u32, multiple: u32) -> u32 {
    if multiple == 0 {
        return value;
    }
    let remainder = value % multiple;
    if remainder == 0 {
        value
    } else {
        value + multiple - remainder
    }
}

/// 一个待识别的文字区域（检测阶段输出的外接矩形）。
#[derive(Debug, Clone, Copy)]
pub struct TextRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// 识别阶段预处理结果（一个 batch）。
pub struct PreprocessedRecBatch {
    pub data: Array4<f32>,
    pub valid_widths: Vec<u32>,
    pub max_width: u32,
}

pub fn preprocess_recognition(
    image: &DynamicImage,
    regions: &[TextRegion],
) -> Result<PreprocessedRecBatch, PreprocessError> {
    let (img_w, img_h) = image.dimensions();
    if img_w == 0 || img_h == 0 {
        return Err(PreprocessError::EmptyImage);
    }
    if regions.is_empty() {
        return Ok(PreprocessedRecBatch {
            data: Array4::<f32>::zeros((0, 3, REC_TARGET_HEIGHT as usize, REC_MAX_WIDTH as usize)),
            valid_widths: Vec::new(),
            max_width: REC_MAX_WIDTH,
        });
    }

    let batch_size = regions.len();
    let target_height = REC_TARGET_HEIGHT;
    let max_width = REC_MAX_WIDTH;

    let mut batch = Array4::<f32>::zeros((
        batch_size,
        3,
        target_height as usize,
        max_width as usize,
    ));

    // padding 区域先统一填成 pad_value=0 归一化后的值
    let pad_normalized = [
        normalize(0.0, NORM_MEAN[0], NORM_STD[0]),
        normalize(0.0, NORM_MEAN[1], NORM_STD[1]),
        normalize(0.0, NORM_MEAN[2], NORM_STD[2]),
    ];
    for sample in 0..batch_size {
        for channel in 0..3 {
            batch
                .slice_mut(ndarray::s![sample, channel, .., ..])
                .fill(pad_normalized[channel]);
        }
    }

    let mut valid_widths = Vec::with_capacity(batch_size);

    for (index, region) in regions.iter().copied().enumerate() {
        // 越界或零面积的区域一律裁剪到图像范围内，保证鲁棒性
        // （不像原实现那样直接报错中断，拖拽工具场景下更希望"尽量出结果"）。
        let x = region.x.min(img_w.saturating_sub(1));
        let y = region.y.min(img_h.saturating_sub(1));
        let width = region.width.min(img_w - x).max(1);
        let height = region.height.min(img_h - y).max(1);

        let cropped = image.crop_imm(x, y, width, height);
        let aspect_ratio = width as f32 / height as f32;
        let mut target_width = (aspect_ratio * target_height as f32)
            .round()
            .clamp(1.0, max_width as f32) as u32;
        if target_width == 0 {
            target_width = 1;
        }

        let resized = cropped.resize_exact(target_width, target_height, FilterType::Lanczos3);
        let rgb = resized.to_rgb8();

        for yy in 0..target_height as usize {
            for xx in 0..target_width as usize {
                let pixel = rgb.get_pixel(xx as u32, yy as u32);
                for channel in 0..3 {
                    let value = pixel[channel] as f32 / 255.0;
                    let normalized = normalize(value, NORM_MEAN[channel], NORM_STD[channel]);
                    batch[[index, channel, yy, xx]] = normalized;
                }
            }
        }

        valid_widths.push(target_width);
        let _ = index;
    }

    Ok(PreprocessedRecBatch {
        data: batch,
        valid_widths,
        max_width,
    })
}

fn normalize(value: f32, mean: f32, std: f32) -> f32 {
    if std == 0.0 {
        0.0
    } else {
        (value - mean) / std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn solid_image(width: u32, height: u32, value: u8) -> DynamicImage {
        let buffer = ImageBuffer::from_pixel(width, height, Rgb([value, value, value]));
        DynamicImage::ImageRgb8(buffer)
    }

    #[test]
    fn resize_long_side_to_limit() {
        let image = solid_image(1920, 1080, 128);
        let result = preprocess_detection(&image, DET_LIMIT_SIDE_LEN).unwrap();
        assert_eq!(result.resized_dims, (960, 544));
        assert!((result.scale_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn pads_to_multiple_of_32() {
        let image = solid_image(123, 77, 200);
        let result = preprocess_detection(&image, DET_LIMIT_SIDE_LEN).unwrap();
        assert_eq!(result.resized_dims, (128, 96));
    }

    #[test]
    fn recognition_batch_shape() {
        let image = solid_image(200, 100, 100);
        let regions = vec![TextRegion {
            x: 10,
            y: 10,
            width: 80,
            height: 40,
        }];
        let batch = preprocess_recognition(&image, &regions).unwrap();
        assert_eq!(
            batch.data.shape(),
            &[1, 3, REC_TARGET_HEIGHT as usize, REC_MAX_WIDTH as usize]
        );
        assert_eq!(batch.valid_widths.len(), 1);
    }
}
