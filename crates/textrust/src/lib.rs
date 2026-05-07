use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Word,
    Punctuation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    pub text: &'a str,
    pub kind: TokenKind,
}

pub fn wordpunct_tokens(input: &str) -> Vec<Token<'_>> {
    let mut tokens = Vec::new();
    let mut start = None;
    let mut current_kind = None;

    for (idx, ch) in input.char_indices() {
        let kind = char_kind(ch);
        match (start, current_kind, kind) {
            (_, _, None) => {
                if let (Some(start_idx), Some(token_kind)) = (start.take(), current_kind.take()) {
                    tokens.push(Token {
                        text: &input[start_idx..idx],
                        kind: token_kind,
                    });
                }
            }
            (None, None, Some(kind)) => {
                start = Some(idx);
                current_kind = Some(kind);
            }
            (Some(start_idx), Some(active), Some(kind)) if active != kind => {
                tokens.push(Token {
                    text: &input[start_idx..idx],
                    kind: active,
                });
                start = Some(idx);
                current_kind = Some(kind);
            }
            _ => {}
        }
    }

    if let (Some(start_idx), Some(token_kind)) = (start, current_kind) {
        tokens.push(Token {
            text: &input[start_idx..],
            kind: token_kind,
        });
    }
    tokens
}

fn char_kind(ch: char) -> Option<TokenKind> {
    if ch.is_ascii_alphanumeric() || ch == '_' {
        Some(TokenKind::Word)
    } else if ch.is_whitespace() {
        None
    } else {
        Some(TokenKind::Punctuation)
    }
}

pub fn lowercase_word_frequencies(input: &str) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for token in wordpunct_tokens(input) {
        if token.kind == TokenKind::Word {
            *counts.entry(token.text.to_ascii_lowercase()).or_insert(0) += 1;
        }
    }
    counts
}

pub fn word_bigrams(input: &str) -> Vec<(String, String)> {
    let words = wordpunct_tokens(input)
        .into_iter()
        .filter(|token| token.kind == TokenKind::Word)
        .map(|token| token.text.to_ascii_lowercase())
        .collect::<Vec<_>>();
    words
        .windows(2)
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect()
}

pub fn token_checksum(tokens: &[Token<'_>]) -> u64 {
    let mut out = 0u64;
    for (idx, token) in tokens.iter().enumerate() {
        out = out.wrapping_add((idx as u64 + 1) * token.text.len() as u64);
        out = out.wrapping_add(match token.kind {
            TokenKind::Word => 17,
            TokenKind::Punctuation => 31,
        });
    }
    out
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
    fn tokenizes_words_and_punctuation_like_wordpunct() {
        let tokens = wordpunct_tokens("Hello, Rust-world! x_1");
        let parts = tokens
            .iter()
            .map(|token| (token.text, token.kind))
            .collect::<Vec<_>>();
        assert_eq!(
            parts,
            vec![
                ("Hello", TokenKind::Word),
                (",", TokenKind::Punctuation),
                ("Rust", TokenKind::Word),
                ("-", TokenKind::Punctuation),
                ("world", TokenKind::Word),
                ("!", TokenKind::Punctuation),
                ("x_1", TokenKind::Word),
            ]
        );
    }

    #[test]
    fn groups_punctuation_runs() {
        let tokens = wordpunct_tokens("wait... what?!");
        let parts = tokens.iter().map(|token| token.text).collect::<Vec<_>>();
        assert_eq!(parts, vec!["wait", "...", "what", "?!"]);
    }

    #[test]
    fn counts_lowercase_word_frequencies() {
        let counts = lowercase_word_frequencies("Alpha beta ALPHA!");
        assert_eq!(counts.get("alpha"), Some(&2));
        assert_eq!(counts.get("beta"), Some(&1));
        assert!(!counts.contains_key("!"));
    }

    #[test]
    fn builds_word_bigrams() {
        let bigrams = word_bigrams("One, two three.");
        assert_eq!(
            bigrams,
            vec![
                ("one".to_string(), "two".to_string()),
                ("two".to_string(), "three".to_string())
            ]
        );
    }
}
