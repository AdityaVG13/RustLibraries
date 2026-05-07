# ImageRust

ImageRust is the image-processing crate in this workspace.

Current scope:

- PPM P6 decoding for 8-bit RGB images.
- RGB to grayscale conversion.
- Nearest-neighbor grayscale resizing.
- Thresholding.
- Same-data benchmark against Pillow.

It is not a full Pillow or scikit-image replacement yet. PNG/JPEG/WebP decoding, color management, metadata, filtering, morphology, segmentation, and advanced resampling remain roadmap items.
