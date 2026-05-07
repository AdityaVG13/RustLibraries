# ImageRust vs Pillow

Same-data local benchmark for the implemented PPM grayscale, resize, and threshold slice.
This is not a full Pillow replacement claim.

- Pillow version: `12.2.0`
- ImageRust wins: 1
- Pillow wins: 0
- Checksum failures: 0
- Global Pillow replacement claim: false

| Case | ImageRust ms | Pillow ms | Speedup | Winner | Checksum |
| --- | ---: | ---: | ---: | --- | --- |
| `ppm_grayscale_resize_threshold_2048x1024` | 1.303 | 21.544 | 16.53x | imagerust | ok |
