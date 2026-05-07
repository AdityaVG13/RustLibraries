use mediaextractrust::{bench_median_ms, extract, MediaKind};

fn html_doc(repetitions: usize) -> Vec<u8> {
    let mut out = String::from("<html><body>");
    for idx in 0..repetitions {
        out.push_str("<section><h2>Invoice ");
        out.push_str(&idx.to_string());
        out.push_str("</h2><p>Total &amp; tax for North&nbsp;America segment ");
        out.push_str(&(idx % 17).to_string());
        out.push_str("</p></section>");
    }
    out.push_str("</body></html>");
    out.into_bytes()
}

fn pdf_doc(repetitions: usize) -> Vec<u8> {
    let mut out = String::from("%PDF-1.4\n1 0 obj <<>> endobj\nstream\nBT\n");
    for idx in 0..repetitions {
        out.push_str("(Invoice ");
        out.push_str(&idx.to_string());
        out.push_str(" total tax segment ");
        out.push_str(&(idx % 17).to_string());
        out.push_str(") Tj\n");
    }
    out.push_str("ET\nendstream\n%%EOF\n");
    out.into_bytes()
}

fn main() {
    let html = html_doc(20_000);
    let pdf = pdf_doc(20_000);
    let html_warmup = extract(MediaKind::Html, &html).unwrap();
    let pdf_warmup = extract(MediaKind::Pdf, &pdf).unwrap();
    let html_ms = bench_median_ms(9, || {
        std::hint::black_box(extract(MediaKind::Html, &html).unwrap());
    });
    let pdf_ms = bench_median_ms(9, || {
        std::hint::black_box(extract(MediaKind::Pdf, &pdf).unwrap());
    });
    println!(
        "{{\"engine\":\"mediaextractrust\",\"cases\":[{{\"name\":\"html_text_20000_sections\",\"millis\":{html_ms:.6},\"checksum\":{}}},{{\"name\":\"pdf_literal_text_20000_lines\",\"millis\":{pdf_ms:.6},\"checksum\":{}}}]}}",
        html_warmup.checksum, pdf_warmup.checksum
    );
}
