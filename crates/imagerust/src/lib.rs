use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    InvalidHeader,
    UnsupportedMaxValue(u32),
    UnexpectedEof,
    DimensionOverflow,
    InvalidResizeTarget,
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeader => write!(f, "invalid PPM header"),
            Self::UnsupportedMaxValue(value) => {
                write!(
                    f,
                    "only 8-bit PPM images are supported, got max value {value}"
                )
            }
            Self::UnexpectedEof => write!(f, "unexpected end of image data"),
            Self::DimensionOverflow => write!(f, "image dimensions overflow addressable memory"),
            Self::InvalidResizeTarget => write!(f, "resize target dimensions must be non-zero"),
        }
    }
}

impl Error for ImageError {}

pub type Result<T> = std::result::Result<T, ImageError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl RgbImage {
    pub fn from_rgb(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(3))
            .ok_or(ImageError::DimensionOverflow)?;
        if pixels.len() != expected {
            return Err(ImageError::UnexpectedEof);
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn decode_ppm(bytes: &[u8]) -> Result<Self> {
        let mut parser = PpmParser { bytes, pos: 0 };
        let magic = parser.next_token().ok_or(ImageError::InvalidHeader)?;
        if magic != b"P6" {
            return Err(ImageError::InvalidHeader);
        }
        let width = parse_usize(parser.next_token().ok_or(ImageError::InvalidHeader)?)?;
        let height = parse_usize(parser.next_token().ok_or(ImageError::InvalidHeader)?)?;
        let max_value = parse_u32(parser.next_token().ok_or(ImageError::InvalidHeader)?)?;
        if max_value != 255 {
            return Err(ImageError::UnsupportedMaxValue(max_value));
        }
        parser.skip_one_ascii_whitespace();
        let expected = width
            .checked_mul(height)
            .and_then(|value| value.checked_mul(3))
            .ok_or(ImageError::DimensionOverflow)?;
        if parser.pos + expected > bytes.len() {
            return Err(ImageError::UnexpectedEof);
        }
        Self::from_rgb(
            width,
            height,
            bytes[parser.pos..parser.pos + expected].to_vec(),
        )
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn grayscale(&self) -> GrayImage {
        let mut pixels = Vec::with_capacity(self.width * self.height);
        for rgb in self.pixels.chunks_exact(3) {
            let r = u32::from(rgb[0]);
            let g = u32::from(rgb[1]);
            let b = u32::from(rgb[2]);
            pixels.push(((77 * r + 150 * g + 29 * b + 128) >> 8) as u8);
        }
        GrayImage {
            width: self.width,
            height: self.height,
            pixels,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrayImage {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl GrayImage {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn resize_nearest(&self, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidResizeTarget);
        }
        let mut pixels = vec![0u8; width * height];
        for y in 0..height {
            let src_y = y * self.height / height;
            let src_row = src_y * self.width;
            let dst_row = y * width;
            for x in 0..width {
                let src_x = x * self.width / width;
                pixels[dst_row + x] = self.pixels[src_row + src_x];
            }
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn threshold(&self, threshold: u8) -> Self {
        let pixels = self
            .pixels
            .iter()
            .map(|value| if *value >= threshold { 255 } else { 0 })
            .collect();
        Self {
            width: self.width,
            height: self.height,
            pixels,
        }
    }

    pub fn checksum(&self) -> u64 {
        self.pixels
            .iter()
            .enumerate()
            .fold(0u64, |acc, (idx, value)| {
                acc.wrapping_add((idx as u64 + 1) * u64::from(*value))
            })
    }
}

struct PpmParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> PpmParser<'a> {
    fn next_token(&mut self) -> Option<&'a [u8]> {
        self.skip_ws_and_comments();
        if self.pos >= self.bytes.len() {
            return None;
        }
        let start = self.pos;
        while self.pos < self.bytes.len() && !self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        Some(&self.bytes[start..self.pos])
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos < self.bytes.len() && self.bytes[self.pos] == b'#' {
                while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            break;
        }
    }

    fn skip_one_ascii_whitespace(&mut self) {
        if self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }
}

fn parse_usize(bytes: &[u8]) -> Result<usize> {
    let mut value = 0usize;
    if bytes.is_empty() {
        return Err(ImageError::InvalidHeader);
    }
    for byte in bytes {
        if !byte.is_ascii_digit() {
            return Err(ImageError::InvalidHeader);
        }
        value = value
            .checked_mul(10)
            .and_then(|value| value.checked_add((byte - b'0') as usize))
            .ok_or(ImageError::DimensionOverflow)?;
    }
    Ok(value)
}

fn parse_u32(bytes: &[u8]) -> Result<u32> {
    parse_usize(bytes).map(|value| value as u32)
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
    fn decodes_ppm_with_comments() {
        let bytes = b"P6\n# generated\n2 1\n255\n\xff\x00\x00\x00\x80\x00";
        let image = RgbImage::decode_ppm(bytes).unwrap();
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 1);
        assert_eq!(image.pixels(), &[255, 0, 0, 0, 128, 0]);
    }

    #[test]
    fn converts_to_grayscale() {
        let image = RgbImage::from_rgb(2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap();
        assert_eq!(image.grayscale().pixels(), &[77, 149]);
    }

    #[test]
    fn resizes_nearest_neighbor() {
        let image = GrayImage {
            width: 2,
            height: 2,
            pixels: vec![1, 2, 3, 4],
        };
        let resized = image.resize_nearest(4, 2).unwrap();
        assert_eq!(resized.pixels(), &[1, 1, 2, 2, 3, 3, 4, 4]);
    }

    #[test]
    fn thresholds_grayscale() {
        let image = GrayImage {
            width: 4,
            height: 1,
            pixels: vec![0, 127, 128, 255],
        };
        assert_eq!(image.threshold(128).pixels(), &[0, 0, 255, 255]);
    }

    #[test]
    fn rejects_bad_payload_length() {
        let err = RgbImage::decode_ppm(b"P6\n2 1\n255\n\x00").unwrap_err();
        assert_eq!(err, ImageError::UnexpectedEof);
    }
}
