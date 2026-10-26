# Data

Section 7 of the paper uses the NIST Biometric Scores Set Release 1 (BSSR1), which is publicly available from
<https://www.nist.gov/itl/iad/btg/nist-biometric-scores-set-bssr1> and is not redistributed here.

Download `bssr1.zip` (178 MB, MD5 `43f40a6383c8b7a485f17dcfc98ad7bb`) and pass it once to the reproduction
script, from the `code/` folder:

```bash
python plbound.py --bssr1 path/to/bssr1.zip
```

The scores are converted into `bssr1_set1.npz`, `bssr1_set2_face.npz` and `bssr1_set3_finger.npz` in this
folder, which later runs of `python plbound.py` read directly. The folder the zip file was extracted to can be
passed instead of the zip file.
