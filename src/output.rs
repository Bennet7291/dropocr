//! 输出相关：时间戳文件名生成、OCR 结果转文本。
//!
//! 时间戳特意不引入 `chrono`/`time` 这类日期时间 crate——
//! 我们只需要"本地时间的年月日时分秒"这一个能力，为了把最终
//! 二进制体积压到 20MB 以内的预算，这里手写一个极简的
//! Gregorian 日历换算（比引入完整时区数据库的 crate 轻量得多）。

use crate::ocr::OcrLine;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// 生成 `yyyymmdd-hhmmss` 形式的文件名基础部分（不含扩展名）。
///
/// 使用**本地时间**：先取系统时区偏移（Windows API），
/// 再做日历换算。若获取时区失败，退化为 UTC。
pub fn timestamp_filename_base() -> String {
    let unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let offset_seconds = local_utc_offset_seconds();
    let local_seconds = unix_seconds + offset_seconds as i64;

    let (year, month, day, hour, minute, second) = civil_from_unix_seconds(local_seconds);

    format!(
        "{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}"
    )
}

/// 把 Unix 时间戳（秒）拆解为本地日历的 年/月/日/时/分/秒。
///
/// 算法：Howard Hinnant 的 `civil_from_days` 公式（公开的、被广泛验证过的
/// 无分支 Gregorian 日历换算算法），只依赖整数运算，不需要任何第三方库。
fn civil_from_unix_seconds(total_seconds: i64) -> (i32, u32, u32, u32, u32, u32) {
    let days = total_seconds.div_euclid(86400);
    let secs_of_day = total_seconds.rem_euclid(86400);

    let hour = (secs_of_day / 3600) as u32;
    let minute = ((secs_of_day % 3600) / 60) as u32;
    let second = (secs_of_day % 60) as u32;

    let (year, month, day) = civil_from_days(days);
    (year, month, day, hour, minute, second)
}

/// Howard Hinnant's `civil_from_days`:
/// https://howardhinnant.github.io/date_algorithms.html#civil_from_days
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };

    (y as i32, m, d)
}

/// 获取本地时区相对 UTC 的偏移秒数。
///
/// 直接 FFI 调用 Win32 `GetTimeZoneInformation`（位于系统自带的
/// `kernel32.dll`），不引入 `winapi`/`windows` 等第三方绑定 crate——
/// Windows 系统本身必带这个 DLL，用 `extern "system"` 手写声明调用，
/// 对最终二进制体积零增量，且不增加依赖树。
///
/// 非 Windows 平台（比如本地用 Linux/macOS 编译调试逻辑时）走 UTC
/// 兜底，只影响开发调试体验，不影响最终 Windows 产物的正确性。
#[cfg(windows)]
fn local_utc_offset_seconds() -> i32 {
    // Win32 `TIME_ZONE_INFORMATION` 结构体布局：
    // https://learn.microsoft.com/windows/win32/api/timezoneapi/ns-timezoneapi-time_zone_information
    #[repr(C)]
    struct SystemTimeWin {
        year: u16,
        month: u16,
        day_of_week: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
        milliseconds: u16,
    }

    #[repr(C)]
    struct TimeZoneInformation {
        bias: i32,
        standard_name: [u16; 32],
        standard_date: SystemTimeWin,
        standard_bias: i32,
        daylight_name: [u16; 32],
        daylight_date: SystemTimeWin,
        daylight_bias: i32,
    }

    extern "system" {
        fn GetTimeZoneInformation(lp_time_zone_information: *mut TimeZoneInformation) -> u32;
    }

    // TIME_ZONE_ID_INVALID
    const TIME_ZONE_ID_INVALID: u32 = 0xFFFF_FFFF;
    const TIME_ZONE_ID_DAYLIGHT: u32 = 2;

    unsafe {
        let mut info: TimeZoneInformation = std::mem::zeroed();
        let result = GetTimeZoneInformation(&mut info as *mut _);
        if result == TIME_ZONE_ID_INVALID {
            return 0;
        }
        // 官方文档（learn.microsoft.com/windows/win32/api/timezoneapi/
        // ns-timezoneapi-time_zone_information）明确写着：
        //   "StandardBias ... This value is added to the value of the
        //   Bias member to form the bias used during standard time."
        //   "DaylightBias ... This value is added to the value of the
        //   Bias member to form the bias used during daylight time."
        // 即：处于夏令时用 bias+daylight_bias，处于标准时间用
        // bias+standard_bias——两者都要叠加各自的修正值，不是只有
        // 夏令时才需要调整。多数地区 standard_bias 恰好是 0（文档原话
        // "In most time zones, the value of this member is zero"），
        // 所以这处修正在大多数机器上不会改变结果，但对极少数
        // standard_bias 非零的地区，之前遗漏这项会得到错误的本地时间。
        // TIME_ZONE_ID_UNKNOWN（该地区从不使用夏令时，如中国）没有
        // 夏令时/标准时间的区分，同样按"标准时间"处理，叠加
        // standard_bias。
        let mut bias_minutes = info.bias + info.standard_bias;
        if result == TIME_ZONE_ID_DAYLIGHT {
            bias_minutes = info.bias + info.daylight_bias;
        }
        -bias_minutes * 60
    }
}

#[cfg(not(windows))]
fn local_utc_offset_seconds() -> i32 {
    // 非 Windows 平台仅用于本地开发调试逻辑正确性，不影响最终
    // Windows exe 的行为（该分支不会被 Windows 编译目标使用）。
    0
}

/// 把一次 OCR 识别结果格式化为最终写入 txt 的文本内容。
pub fn format_ocr_result(source_path: &Path, lines: &[OcrLine]) -> String {
    if lines.is_empty() {
        return format!(
            "未识别到文字\n来源文件: {}\n",
            source_path.display()
        );
    }

    let mut content = String::new();
    for line in lines {
        content.push_str(&line.text);
        content.push('\n');
    }
    content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_from_days_epoch() {
        // 1970-01-01 是 days=0
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }

    #[test]
    fn civil_from_days_known_date() {
        // 2024-01-01 距 1970-01-01 是 19723 天（含 13 个闰年：
        // 1972,76,80,84,88,92,96,2000,04,08,12,16,20 共 13 个）
        // 54*365 + 13 = 19710+13=19723
        assert_eq!(civil_from_days(19723), (2024, 1, 1));
    }

    #[test]
    fn civil_from_days_leap_day() {
        // 2024-02-29 (闰年)
        let days_to_2024_02_29 = 19723 + 31 + 28; // 1月31天 + 2月到29号前28天
        assert_eq!(civil_from_days(days_to_2024_02_29), (2024, 2, 29));
    }

    #[test]
    fn civil_from_days_century_leap_year() {
        // 2000 年能被 400 整除，是闰年（"能被100整除但不能被400整除的
        // 世纪年不是闰年"这条 Gregorian 规则最容易被业余实现写错的
        // 就是 2000 这种能被400整除的特例）。11016 是用 Python
        // datetime 独立算出的 1970-01-01 到 2000-02-29 的天数，
        // 交叉验证过，不是凭手算。
        assert_eq!(civil_from_days(11016), (2000, 2, 29));
    }

    #[test]
    fn civil_from_days_century_non_leap_year() {
        // 1900 年能被 100 整除但不能被 400 整除，不是闰年，
        // 所以 1900 年 2 月只有 28 天，2月的最后一天的次日应该
        // 直接跳到 3月1日（而不是有 29 号）。-25508 同样用
        // Python datetime 独立算出并交叉验证过（1970-01-01 往前数）。
        assert_eq!(civil_from_days(-25508), (1900, 3, 1));
    }

    #[test]
    fn timestamp_format_shape() {
        let name = timestamp_filename_base();
        // 形如 20260813-143022，长度固定为 15（8位日期 + '-' + 6位时间）
        assert_eq!(name.len(), 15);
        assert!(name.chars().nth(8) == Some('-'));
    }
}
