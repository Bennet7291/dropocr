//! 识别字典：PaddleOCR CTC 头部使用的字符表。
//!
//! 关键约定（已对照 PaddleOCR 官方源码
//! `ppocr/postprocess/rec_postprocess.py::CTCLabelDecode.add_special_char`
//! 核实过）：blank token 固定插入在字典**最前面**，索引为 0：
//! ```python
//! def add_special_char(self, dict_character):
//!     dict_character = ['blank'] + dict_character
//!     return dict_character
//! ```
//! 因此本模块不从磁盘文件读取字典，而是把字典内容通过
//! `include_str!` 直接编译进最终二进制，运行期零文件依赖。

use std::collections::HashMap;

/// 字典加载错误。
#[derive(Debug)]
pub enum DictionaryError {
    /// 字典内容为空（不含任何有效 token）。
    Empty,
    /// 字典内出现重复 token，说明词表文件本身有问题。
    Duplicate { line_number: usize, token: String },
}

impl std::fmt::Display for DictionaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DictionaryError::Empty => write!(f, "字典内容为空"),
            DictionaryError::Duplicate { line_number, token } => {
                write!(f, "字典第 {line_number} 行出现重复 token: {token:?}")
            }
        }
    }
}

impl std::error::Error for DictionaryError {}

/// 识别字典：索引 <-> 字符 双向映射。
#[derive(Debug, Clone)]
pub struct RecDictionary {
    tokens: Vec<String>,
}

impl RecDictionary {
    /// 从内嵌的字典文本（每行一个 token）构造。
    ///
    /// 与 PaddleOCR 官方约定一致：blank 固定为索引 0，
    /// 随后按文件行序依次编号。
    pub fn from_embedded_str(contents: &str) -> Result<Self, DictionaryError> {
        let mut tokens = Vec::new();
        let mut seen: HashMap<&str, usize> = HashMap::new();

        // blank 永远是索引 0。
        tokens.push("blank".to_string());

        for (line_number, raw_line) in contents.lines().enumerate() {
            // 去掉可能存在的 UTF-8 BOM（第一行）。
            let line = if line_number == 0 {
                raw_line.trim_start_matches('\u{FEFF}')
            } else {
                raw_line
            };

            // PaddleOCR 字典文件里，一行如果只包含一个空格，
            // 这个空格本身就是有效字符（代表识别结果中的空格），
            // 因此不能无脑 trim 掉整行内容。
            // 规则：只有整行为空字符串（长度为 0）才跳过；
            // 否则保留原始内容（不 trim 两端空白）。
            if line.is_empty() {
                continue;
            }

            if let Some(&prev_index) = seen.get(line) {
                let _ = prev_index; // 仅用于说明重复来源，避免未使用告警
                return Err(DictionaryError::Duplicate {
                    line_number: line_number + 1,
                    token: line.to_string(),
                });
            }

            seen.insert(line, tokens.len());
            tokens.push(line.to_string());
        }

        if tokens.len() <= 1 {
            return Err(DictionaryError::Empty);
        }

        Ok(Self { tokens })
    }

    /// 字典总长度（含 blank）。
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// blank token 的索引，固定为 0。
    pub fn blank_id(&self) -> usize {
        0
    }

    /// 根据索引取字符；越界返回 None。
    pub fn token(&self, index: usize) -> Option<&str> {
        self.tokens.get(index).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_is_index_zero() {
        let dict = RecDictionary::from_embedded_str("a\nb\nc\n").unwrap();
        assert_eq!(dict.blank_id(), 0);
        assert_eq!(dict.token(0), Some("blank"));
        assert_eq!(dict.token(1), Some("a"));
        assert_eq!(dict.token(2), Some("b"));
        assert_eq!(dict.token(3), Some("c"));
        assert_eq!(dict.len(), 4);
    }

    #[test]
    fn preserves_space_only_line() {
        // PaddleOCR 字典里常见一行只有一个空格，代表"空格"字符本身。
        let dict = RecDictionary::from_embedded_str(" \nalpha\n").unwrap();
        assert_eq!(dict.token(1), Some(" "));
        assert_eq!(dict.token(2), Some("alpha"));
    }

    #[test]
    fn rejects_duplicate_token() {
        let err = RecDictionary::from_embedded_str("foo\nbar\nfoo\n").unwrap_err();
        match err {
            DictionaryError::Duplicate { line_number, .. } => assert_eq!(line_number, 3),
            _ => panic!("expected Duplicate error"),
        }
    }

    #[test]
    fn rejects_empty_dictionary() {
        let err = RecDictionary::from_embedded_str("\n\n\n").unwrap_err();
        assert!(matches!(err, DictionaryError::Empty));
    }
}
