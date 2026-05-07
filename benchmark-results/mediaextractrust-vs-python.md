# MediaExtractRust vs Python Extraction Libraries

Same-data local benchmark for the implemented HTML and uncompressed PDF text extraction slice.
This is not a full OCR, Office document, or general document pipeline replacement claim.

- MediaExtractRust wins: 2
- Python wins: 0
- Checksum failures: 0
- Global Python document pipeline replacement claim: false

| Case | MediaExtractRust ms | Python ms | Speedup | Winner | Checksum |
| --- | ---: | ---: | ---: | --- | --- |
| `html_text_20000_sections` | 5.149 | 418.185 | 81.21x | mediaextractrust | ok |
| `pdf_literal_text_20000_lines` | 3.829 | 470.740 | 122.93x | mediaextractrust | ok |
