// 关键：告诉 Windows 这是一个 GUI 子系统程序，不分配/弹出控制台窗口。
// 这样双击或拖拽到 exe 上运行时，不会闪出一个黑色 cmd 窗口。
// 该属性只在 Windows target 下生效，其他平台忽略（不影响非 Windows 编译）。
#![windows_subsystem = "windows"]

mod ocr;
mod output;

use image::DynamicImage;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// 程序支持处理的图片扩展名（不区分大小写）。
/// avif / jxl 暂未接入解码器，先占位声明，后续补上后只需要
/// 在这里加扩展名 + 在 `decode_image` 里加对应分支。
const SUPPORTED_EXTENSIONS: &[&str] = &["webp", "jpeg", "jpg", "png"];

fn main() {
    // 拖拽到 exe 上时，Windows 会把每个被拖拽的文件路径作为一个独立的
    // 命令行参数传入（argv[0] 是 exe 自身路径，从 argv[1] 开始才是文件）。
    // 不需要任何参数解析库：这就是全部输入形式。
    //
    // 特意用 args_os() 而非 args()：后者遇到非合法 UTF-8 的参数会直接
    // panic 整个进程（这是 Rust 标准库文档明确记录的行为），对一个靠
    // 拖拽任意文件路径驱动的工具来说不能接受一条边缘情况路径就让整个
    // 程序崩溃、一个文件都处理不了。args_os() 保留原始 OsString，
    // 永远不会因为编码问题 panic，PathBuf 可以直接从 OsString 构造。
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();

    if args.is_empty() {
        // 没有拖入任何文件（比如用户直接双击 exe）。
        // 没有控制台可以提示，唯一能做的是在 exe 所在目录留一个
        // 简短的说明文件，避免用户完全摸不着头脑。
        write_usage_hint();
        return;
    }

    // 引擎只需要初始化一次（加载模型是相对耗时的操作），
    // 拖入多个文件时复用同一个引擎实例。
    let engine = match ocr::OcrEngine::new() {
        Ok(engine) => engine,
        Err(err) => {
            write_fatal_error(&format!("OCR 引擎初始化失败: {err}"));
            return;
        }
    };

    // 输出目录：exe 所在目录（而不是当前工作目录，两者在某些拖拽场景下可能不同）。
    let output_dir = exe_directory();

    // 同一批次拖拽的多个文件，时间戳可能落在同一秒，
    // 用一个已用文件名集合做去重，避免互相覆盖。
    let mut used_filenames: Vec<String> = Vec::new();

    for arg in &args {
        // Path::new 接受任何 AsRef<OsStr> 的类型（含 &OsString），
        // 不需要经过 String 转换，天然兼容任意合法文件路径。
        let path = Path::new(arg);
        process_one_file(path, &engine, &output_dir, &mut used_filenames);
    }
}

fn process_one_file(
    path: &Path,
    engine: &ocr::OcrEngine,
    output_dir: &Path,
    used_filenames: &mut Vec<String>,
) {
    if !is_supported_image(path) {
        // 拖进来的文件不是受支持的图片格式：同样不弹窗提示，
        // 而是生成一个对应的说明 txt，让用户能在目录里看到处理结果。
        let message = format!(
            "文件未处理：不支持的格式或找不到文件\n路径: {}\n\n目前支持的格式: {}\n",
            path.display(),
            SUPPORTED_EXTENSIONS.join(", ")
        );
        write_result_file(output_dir, used_filenames, &message);
        return;
    }

    let image = match decode_image(path) {
        Ok(image) => image,
        Err(err) => {
            let message = format!(
                "文件未处理：图片解码失败\n路径: {}\n错误: {err}\n",
                path.display()
            );
            write_result_file(output_dir, used_filenames, &message);
            return;
        }
    };

    match engine.recognize(&image) {
        Ok(lines) => {
            let content = output::format_ocr_result(path, &lines);
            write_result_file(output_dir, used_filenames, &content);
        }
        Err(err) => {
            let message = format!(
                "文件处理失败：OCR 识别出错\n路径: {}\n错误: {err}\n",
                path.display()
            );
            write_result_file(output_dir, used_filenames, &message);
        }
    }
}

/// 判断文件是否是受支持的图片格式（按扩展名）。
fn is_supported_image(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let lower = ext.to_ascii_lowercase();
            SUPPORTED_EXTENSIONS.contains(&lower.as_str())
        })
        .unwrap_or(false)
}

/// 解码图片文件为统一的 DynamicImage。
fn decode_image(path: &Path) -> Result<DynamicImage, image::ImageError> {
    // `image::open` 会按文件内容嗅探真实格式（而不仅是看扩展名），
    // 对扩展名和实际内容不一致的文件也有一定容错性。
    image::open(path)
}

/// exe 所在目录；获取失败时退化为当前工作目录。
fn exe_directory() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// 生成形如 `yyyymmdd-hhmmss.txt` 的文件名，若同名已存在（同一秒内多个文件）
/// 则追加 `-1`、`-2` 等后缀。
fn write_result_file(output_dir: &Path, used_filenames: &mut Vec<String>, content: &str) {
    let base_name = output::timestamp_filename_base();
    let mut candidate = format!("{base_name}.txt");
    let mut suffix = 1u32;

    while used_filenames.contains(&candidate) || output_dir.join(&candidate).exists() {
        candidate = format!("{base_name}-{suffix}.txt");
        suffix += 1;
    }

    let full_path = output_dir.join(&candidate);
    // 写入失败（比如目录不可写）也没有地方能提示用户，
    // 这里静默忽略——已经是无 GUI/CLI 工具能做的极限了。
    let _ = std::fs::write(&full_path, content);

    used_filenames.push(candidate);
}

/// 无参数运行时（没有拖入任何文件）留下的说明文件。
fn write_usage_hint() {
    let output_dir = exe_directory();
    let mut used = Vec::new();
    let content = "使用方法：把图片文件拖拽到本 exe 上即可自动识别文字并生成同目录下的 txt 文件。\n\
支持格式：webp, jpeg, jpg, png\n";
    write_result_file(&output_dir, &mut used, content);
}

fn write_fatal_error(message: &str) {
    let output_dir = exe_directory();
    let mut used = Vec::new();
    write_result_file(&output_dir, &mut used, message);
}
