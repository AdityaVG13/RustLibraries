# TextRust

TextRust is the text-processing crate in this workspace. The Rust crate is named `textrust`.

Current scope:

- NLTK-style word/punctuation tokenization for ASCII text.
- Lowercase word-frequency counts.
- Word bigrams.
- Same-data benchmark against NLTK `wordpunct_tokenize`.

It is not a full NLTK or spaCy replacement yet. Unicode segmentation, sentence splitting, stemming, lemmatization, entity pipelines, vectorization, language models, and training workflows remain roadmap items.
