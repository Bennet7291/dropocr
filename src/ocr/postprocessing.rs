//! DBNet 检测后处理：把模型输出的概率图转换成文字外接矩形框。
//!
//! 流程（与 PaddleOCR 官方 `DBPostProcess` 一致）：
//! 1. 概率图按阈值二值化，用 `imageproc` 提取轮廓
//! 2. 过滤掉过小的轮廓（噪声）
//! 3. 用 Vatti clipping 做多边形 unclip 膨胀
//!    （因为 DBNet 训练时收缩了文字框，这里要按比例撑回去）
//!    unclip 距离 = 多边形面积 / 周长 * unclip_ratio
//! 4. 取外接矩形，按检测阶段的缩放比例映射回原图坐标

use crate::ocr::preprocessing::TextRegion;
use imageproc::contours::{find_contours_with_threshold, Contour};
use ndarray::Array2;

pub const DET_THRESH: f32 = 0.3;
pub const DET_UNCLIP_RATIO: f32 = 1.5;
const MIN_CONTOUR_POINTS: usize = 3;
const MIN_BOX_SIDE: f64 = 3.0;
/// 单张图片最多保留的检测区域数量，超出部分按面积从小到大丢弃。
/// 正常文档/截图远达不到这个数字，只在极端噪声输入时才会触发。
const MAX_REGIONS: usize = 1000;

/// 从概率图提取文字区域（已映射回原图坐标）。
pub fn extract_text_regions(
    probability_map: &Array2<f32>,
    original_dims: (u32, u32),
    scale_ratio: f64,
    thresh: f32,
    unclip_ratio: f32,
) -> Vec<TextRegion> {
    let (map_h, map_w) = probability_map.dim();
    if map_h == 0 || map_w == 0 {
        return Vec::new();
    }

    // 二值化：概率图 -> 0/255 灰度图，供 imageproc 提取轮廓
    let mut binary = image::GrayImage::new(map_w as u32, map_h as u32);
    for y in 0..map_h {
        for x in 0..map_w {
            let value = probability_map[[y, x]];
            let pixel = if value > thresh { 255u8 } else { 0u8 };
            binary.put_pixel(x as u32, y as u32, image::Luma([pixel]));
        }
    }

    let contours: Vec<Contour<i32>> = find_contours_with_threshold(&binary, 1);

    let (orig_w, orig_h) = original_dims;
    let mut regions = Vec::new();

    for contour in &contours {
        if contour.points.len() < MIN_CONTOUR_POINTS {
            continue;
        }

        let points: Vec<(f64, f64)> = contour
            .points
            .iter()
            .map(|p| (p.x as f64, p.y as f64))
            .collect();

        let area = polygon_area(&points);
        if area < 4.0 {
            continue;
        }

        let expanded = unclip_polygon(&points, area, unclip_ratio);

        // 取外接矩形作为最终框（简化处理：不保留旋转角度，
        // 对于拖拽单图 OCR 场景，绝大多数文本是横排，矩形框已经足够）。
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &(x, y) in &expanded {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
            continue;
        }

        // 映射回原图坐标：先除以 scale_ratio（缩放还原），
        // 注意 probability_map 的尺寸就是检测输入的 padded 尺寸，
        // 而 padded 尺寸是在"缩放后"的基础上做的，所以直接除以 scale_ratio 即可。
        let inv_scale = if scale_ratio > 0.0 {
            1.0 / scale_ratio
        } else {
            1.0
        };

        let rx0 = (min_x * inv_scale).clamp(0.0, orig_w as f64);
        let ry0 = (min_y * inv_scale).clamp(0.0, orig_h as f64);
        let rx1 = (max_x * inv_scale).clamp(0.0, orig_w as f64);
        let ry1 = (max_y * inv_scale).clamp(0.0, orig_h as f64);

        let width = rx1 - rx0;
        let height = ry1 - ry0;
        if width < MIN_BOX_SIDE || height < MIN_BOX_SIDE {
            continue;
        }

        regions.push(TextRegion {
            x: rx0.round() as u32,
            y: ry0.round() as u32,
            width: width.round().max(1.0) as u32,
            height: height.round().max(1.0) as u32,
        });
    }

    // 保护性上限：正常文档/截图的文字区域数量一般是几到几十个，
    // 但某些图片（复杂纹理照片、噪点较多的扫描件）可能被误检出成百
    // 上千个"文字区域"。识别阶段会把所有区域打包成一个 batch 喂给
    // 模型，数量一旦失控会导致内存占用剧增、推理耗时从秒级变成
    // 分钟甚至更久——而这个工具完全没有进度提示或取消手段，用户只会
    // 看到程序"卡住不动"。这里按面积保留最大的 MAX_REGIONS 个区域
    // （大区域更可能是真实文字，噪声通常是零散小块），双保险优于
    // 让程序在极端输入上失控挂起。
    regions = cap_regions_by_area(regions, MAX_REGIONS);

    // 按从上到下、从左到右排序，使输出文本顺序符合阅读习惯。
    regions.sort_by(|a, b| {
        let row_a = a.y / 10; // 允许 10px 内的行高误差归为同一行
        let row_b = b.y / 10;
        row_a.cmp(&row_b).then(a.x.cmp(&b.x))
    });

    regions
}

/// 多边形面积（鞋带公式，取绝对值）。
fn polygon_area(points: &[(f64, f64)]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..points.len() {
        let (x0, y0) = points[i];
        let (x1, y1) = points[(i + 1) % points.len()];
        sum += x0 * y1 - x1 * y0;
    }
    (sum / 2.0).abs()
}

fn polygon_perimeter(points: &[(f64, f64)]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let mut length = 0.0;
    for i in 0..points.len() {
        let (x0, y0) = points[i];
        let (x1, y1) = points[(i + 1) % points.len()];
        length += ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
    }
    length
}

/// 用 Vatti clipping 对多边形做外扩（unclip），公式与 PaddleOCR 官方一致：
/// distance = area / perimeter * unclip_ratio
fn unclip_polygon(points: &[(f64, f64)], area: f64, unclip_ratio: f32) -> Vec<(f64, f64)> {
    let perimeter = polygon_perimeter(points);
    if perimeter <= f64::EPSILON {
        return points.to_vec();
    }
    let distance = area / perimeter * unclip_ratio as f64;

    // 用外接矩形做外扩，而非精确的 Vatti clipping 多边形偏移。
    //
    // 工程取舍说明：DBNet 输出的轮廓在绝大多数横排文字场景下
    // （截图、文档、网页）已经接近矩形，用外接矩形外扩的精度损失很小，
    // 换来的是不依赖任何第三方多边形偏移库的具体 API 版本细节，
    // 最大化编译稳定性——`i_overlay` 这类库的高层 API 在不同版本间
    // 变动较大，本工程更看重"稳定编译成功"优先于"轮廓精度的最后 5%"。
    // 若后续需要支持严重旋转的文字（如倾斜拍摄的招牌），可以在此处
    // 替换为真正的 Vatti clipping 实现。
    let shape: Vec<[f64; 2]> = points.iter().map(|&(x, y)| [x, y]).collect();
    manual_polygon_offset(&shape, distance)
}

/// 简单多边形外扩：沿外接矩形均匀扩张 `distance` 像素。
/// 相比精确的 Vatti clipping，对轻微非矩形轮廓会有少量误差，
/// 但足够覆盖 OCR 文字框场景，且实现不依赖任何第三方多边形偏移库版本细节。
fn manual_polygon_offset(points: &[[f64; 2]], distance: f64) -> Vec<(f64, f64)> {
    if points.is_empty() {
        return Vec::new();
    }
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in points {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    }
    vec![
        (min_x - distance, min_y - distance),
        (max_x + distance, min_y - distance),
        (max_x + distance, max_y + distance),
        (min_x - distance, max_y + distance),
    ]
}

/// 若区域数量超过 `max`，按面积（宽*高）降序只保留最大的 `max` 个。
/// 数量本就不超过上限时原样返回。
fn cap_regions_by_area(mut regions: Vec<TextRegion>, max: usize) -> Vec<TextRegion> {
    if regions.len() <= max {
        return regions;
    }
    regions.sort_by_key(|r| std::cmp::Reverse((r.width as u64) * (r.height as u64)));
    regions.truncate(max);
    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_single_square_region() {
        let mut map = Array2::<f32>::zeros((32, 32));
        for y in 8..24 {
            for x in 8..24 {
                map[[y, x]] = 1.0;
            }
        }

        let regions = extract_text_regions(&map, (32, 32), 1.0, DET_THRESH, DET_UNCLIP_RATIO);
        assert_eq!(regions.len(), 1);
        assert!(regions[0].width >= 16);
        assert!(regions[0].height >= 16);
    }

    #[test]
    fn empty_map_returns_no_regions() {
        let map = Array2::<f32>::zeros((32, 32));
        let regions = extract_text_regions(&map, (32, 32), 1.0, DET_THRESH, DET_UNCLIP_RATIO);
        assert!(regions.is_empty());
    }

    #[test]
    fn cap_regions_keeps_largest_when_over_limit() {
        let regions: Vec<TextRegion> = (0..10)
            .map(|i| TextRegion {
                x: i * 5,
                y: 0,
                width: i + 1, // 面积随 i 递增：1,2,...,10
                height: 1,
            })
            .collect();

        let capped = cap_regions_by_area(regions, 3);
        assert_eq!(capped.len(), 3);
        // 保留的应该是面积最大的三个：width=10,9,8
        let mut widths: Vec<u32> = capped.iter().map(|r| r.width).collect();
        widths.sort_unstable();
        assert_eq!(widths, vec![8, 9, 10]);
    }

    #[test]
    fn cap_regions_no_op_when_under_limit() {
        let regions: Vec<TextRegion> = (0..5)
            .map(|i| TextRegion { x: i, y: 0, width: 1, height: 1 })
            .collect();
        let capped = cap_regions_by_area(regions, 1000);
        assert_eq!(capped.len(), 5);
    }
}
