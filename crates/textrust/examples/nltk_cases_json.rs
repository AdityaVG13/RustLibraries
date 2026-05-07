use textrust::{bench_median_ms, token_checksum, wordpunct_tokens};

fn corpus(sentences: usize) -> String {
    let mut out = String::new();
    for idx in 0..sentences {
        out.push_str("Revenue, cost, and margin changed in segment_");
        out.push_str(&(idx % 19).to_string());
        out.push_str("! Model ");
        out.push_str(&(idx % 7).to_string());
        out.push_str(" scored ");
        out.push_str(&(idx % 101).to_string());
        out.push_str(" points; retry? yes.\n");
    }
    out
}

fn main() {
    let corpus = corpus(200_000);
    let warmup = wordpunct_tokens(&corpus);
    let checksum = token_checksum(&warmup);
    let millis = bench_median_ms(9, || {
        let tokens = wordpunct_tokens(&corpus);
        std::hint::black_box(tokens);
    });
    println!(
        "{{\"engine\":\"textrust\",\"cases\":[{{\"name\":\"wordpunct_tokenize_200000_sentences\",\"millis\":{millis:.6},\"checksum\":{checksum}}}]}}"
    );
}
