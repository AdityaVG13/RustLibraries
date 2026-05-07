# TextRust vs NLTK

Same-data local benchmark for the implemented word/punctuation tokenization slice.
This is not a full NLTK or spaCy replacement claim.

- NLTK version: `3.9.4`
- TextRust wins: 1
- NLTK wins: 0
- Checksum failures: 0
- Global NLTK replacement claim: false

| Case | TextRust ms | NLTK ms | Speedup | Winner | Checksum |
| --- | ---: | ---: | ---: | --- | --- |
| `wordpunct_tokenize_200000_sentences` | 22.024 | 526.828 | 23.92x | textrust | ok |
