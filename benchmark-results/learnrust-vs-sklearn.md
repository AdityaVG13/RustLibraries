# LearnRust vs scikit-learn

Same-data local benchmark for the implemented preprocessing, nearest-centroid, and metrics slice.
This is not a full scikit-learn replacement claim.

- scikit-learn version: `1.8.0`
- LearnRust wins: 3
- scikit-learn wins: 0
- Geomean speedup vs scikit-learn: 6.53x
- Checksum failures: 0
- Global scikit-learn replacement claim: false

| Case | LearnRust ms | scikit-learn ms | Speedup | Winner | Checksum |
| --- | ---: | ---: | ---: | --- | --- |
| `standard_scaler_fit_transform_200000x12` | 2.976 | 8.922 | 3.00x | learnrust | ok |
| `nearest_centroid_fit_predict_120000x8_40000x8_6` | 1.056 | 12.185 | 11.53x | learnrust | ok |
| `accuracy_confusion_1000000_8` | 3.663 | 29.475 | 8.05x | learnrust | ok |
