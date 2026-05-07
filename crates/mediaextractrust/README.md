# MediaExtractRust

MediaExtractRust is the document/media extraction crate in this workspace.

Current scope:

- UTF-8 text and Markdown extraction with whitespace normalization.
- HTML tag stripping with common entity decoding.
- Fast extraction of uncompressed PDF literal and hex text strings.
- Same-data benchmarks against Python extraction libraries.

It is not a full replacement for OCR, Office parsing, scanned PDFs, layout recovery, compressed PDF streams, or `unstructured`-style pipelines yet. Those remain roadmap work.
