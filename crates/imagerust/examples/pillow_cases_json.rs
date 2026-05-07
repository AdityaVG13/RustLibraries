use imagerust::{bench_median_ms, RgbImage};

fn ppm_image(width: usize, height: usize) -> Vec<u8> {
    let mut out = format!("P6\n{width} {height}\n255\n").into_bytes();
    out.reserve(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            out.push(((x * 13 + y * 3) & 255) as u8);
            out.push(((x * 7 + y * 11) & 255) as u8);
            out.push(((x * 5 + y * 17) & 255) as u8);
        }
    }
    out
}

fn main() {
    let bytes = ppm_image(2048, 1024);
    let image = RgbImage::decode_ppm(&bytes).unwrap();
    let warmup = image
        .grayscale()
        .resize_nearest(1024, 512)
        .unwrap()
        .threshold(128);
    let checksum = warmup.checksum();
    let millis = bench_median_ms(9, || {
        let out = image
            .grayscale()
            .resize_nearest(1024, 512)
            .unwrap()
            .threshold(128);
        std::hint::black_box(out);
    });
    println!(
        "{{\"engine\":\"imagerust\",\"cases\":[{{\"name\":\"ppm_grayscale_resize_threshold_2048x1024\",\"millis\":{millis:.6},\"checksum\":{checksum}}}]}}"
    );
}
