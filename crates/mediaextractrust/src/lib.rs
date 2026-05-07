use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Text,
    Markdown,
    Html,
    Pdf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractError {
    InvalidUtf8,
    UnsupportedBinaryPdf,
}

impl fmt::Display for ExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUtf8 => write!(f, "input is not valid UTF-8"),
            Self::UnsupportedBinaryPdf => write!(f, "only text-readable PDF streams are supported"),
        }
    }
}

impl Error for ExtractError {}

pub type Result<T> = std::result::Result<T, ExtractError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedText {
    pub text: String,
    pub byte_len: usize,
    pub word_count: usize,
    pub checksum: u64,
}

pub fn extract(kind: MediaKind, bytes: &[u8]) -> Result<ExtractedText> {
    let raw = match kind {
        MediaKind::Text | MediaKind::Markdown => std::str::from_utf8(bytes)
            .map_err(|_| ExtractError::InvalidUtf8)?
            .to_string(),
        MediaKind::Html => extract_html(bytes)?,
        MediaKind::Pdf => extract_pdf_text_literals(bytes)?,
    };
    Ok(summarize(normalize_whitespace(&raw)))
}

pub fn normalize_whitespace(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut pending_space = false;
    for ch in input.chars() {
        if ch.is_whitespace() {
            pending_space = !out.is_empty();
            continue;
        }
        if pending_space {
            out.push(' ');
            pending_space = false;
        }
        out.push(ch);
    }
    out
}

fn summarize(text: String) -> ExtractedText {
    ExtractedText {
        byte_len: text.len(),
        word_count: text.split_whitespace().count(),
        checksum: checksum(text.as_bytes()),
        text,
    }
}

pub fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = 14_695_981_039_346_656_037u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

fn extract_html(bytes: &[u8]) -> Result<String> {
    let input = std::str::from_utf8(bytes).map_err(|_| ExtractError::InvalidUtf8)?;
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    let mut entity = String::new();
    let mut in_entity = false;

    for ch in input.chars() {
        if in_tag {
            if ch == '>' {
                in_tag = false;
                out.push(' ');
            }
            continue;
        }
        if in_entity {
            if ch == ';' {
                out.push_str(decode_entity(&entity));
                entity.clear();
                in_entity = false;
            } else if entity.len() < 12 {
                entity.push(ch);
            } else {
                out.push('&');
                out.push_str(&entity);
                entity.clear();
                in_entity = false;
                out.push(ch);
            }
            continue;
        }
        match ch {
            '<' => in_tag = true,
            '&' => in_entity = true,
            _ => out.push(ch),
        }
    }
    if in_entity {
        out.push('&');
        out.push_str(&entity);
    }
    Ok(out)
}

fn decode_entity(entity: &str) -> &str {
    match entity {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" => " ",
        _ => "",
    }
}

fn extract_pdf_text_literals(bytes: &[u8]) -> Result<String> {
    if bytes.contains(&0) {
        return Err(ExtractError::UnsupportedBinaryPdf);
    }
    let mut out = String::new();
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == b'(' {
            let (literal, next_idx) = parse_pdf_literal(bytes, idx + 1);
            if !literal.is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(&literal);
            }
            idx = next_idx;
        } else if bytes[idx] == b'<' && idx + 1 < bytes.len() && bytes[idx + 1] != b'<' {
            let (literal, next_idx) = parse_pdf_hex_string(bytes, idx + 1);
            if !literal.is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(&literal);
            }
            idx = next_idx;
        } else {
            idx += 1;
        }
    }
    Ok(out)
}

fn parse_pdf_literal(bytes: &[u8], mut idx: usize) -> (String, usize) {
    let mut out = String::new();
    let mut depth = 1usize;
    while idx < bytes.len() {
        match bytes[idx] {
            b'\\' if idx + 1 < bytes.len() => {
                idx += 1;
                match bytes[idx] {
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'(' => out.push('('),
                    b')' => out.push(')'),
                    b'\\' => out.push('\\'),
                    byte => out.push(byte as char),
                }
                idx += 1;
            }
            b'(' => {
                depth += 1;
                out.push('(');
                idx += 1;
            }
            b')' => {
                depth -= 1;
                idx += 1;
                if depth == 0 {
                    return (out, idx);
                }
                out.push(')');
            }
            byte => {
                out.push(byte as char);
                idx += 1;
            }
        }
    }
    (out, idx)
}

fn parse_pdf_hex_string(bytes: &[u8], mut idx: usize) -> (String, usize) {
    let mut hex = Vec::new();
    while idx < bytes.len() {
        match bytes[idx] {
            b'>' => {
                idx += 1;
                break;
            }
            byte if byte.is_ascii_hexdigit() => {
                hex.push(byte);
                idx += 1;
            }
            _ => idx += 1,
        }
    }
    if hex.len() % 2 == 1 {
        hex.push(b'0');
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    for pair in hex.chunks_exact(2) {
        let high = hex_value(pair[0]);
        let low = hex_value(pair[1]);
        out.push((high << 4) | low);
    }
    (String::from_utf8_lossy(&out).into_owned(), idx)
}

fn hex_value(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => 0,
    }
}

pub fn median_duration(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

pub fn bench_median_ms<F>(rounds: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        f();
        samples.push(start.elapsed());
    }
    median_duration(samples).as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_text_and_markdown() {
        let out = extract(MediaKind::Markdown, b"# Title\n\nalpha   beta\tgamma").unwrap();
        assert_eq!(out.text, "# Title alpha beta gamma");
        assert_eq!(out.word_count, 5);
    }

    #[test]
    fn extracts_html_text_and_entities() {
        let out = extract(
            MediaKind::Html,
            b"<main><h1>Revenue &amp; Costs</h1><p>North&nbsp;America</p></main>",
        )
        .unwrap();
        assert_eq!(out.text, "Revenue & Costs North America");
    }

    #[test]
    fn extracts_pdf_literal_strings() {
        let pdf = b"%PDF-1.4\nBT /F1 12 Tf (Hello\\) PDF) Tj [( table) 10 ( text)] TJ ET";
        let out = extract(MediaKind::Pdf, pdf).unwrap();
        assert_eq!(out.text, "Hello) PDF table text");
    }

    #[test]
    fn extracts_pdf_hex_strings() {
        let pdf = b"%PDF-1.4\nBT <48656c6c6f> Tj ET";
        let out = extract(MediaKind::Pdf, pdf).unwrap();
        assert_eq!(out.text, "Hello");
    }

    #[test]
    fn rejects_binary_pdf_payloads() {
        let err = extract(MediaKind::Pdf, b"%PDF\x00binary").unwrap_err();
        assert_eq!(err, ExtractError::UnsupportedBinaryPdf);
    }
}
