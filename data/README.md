# Data

Section 8 of the paper uses the NIST Biometric Scores Set Release 1 (BSSR1), which is publicly available from
<https://www.nist.gov/itl/iad/btg/nist-biometric-scores-set-bssr1> and is not redistributed here.

1. Download `bssr1.zip` (178 MB, MD5 `43f40a6383c8b7a485f17dcfc98ad7bb`) and unzip it.
2. From the `code/` folder run

   ```bash
   python prepare_bssr1.py <dir>/bssr1/fing_x_face ../data/bssr1_set1.npz
   python prepare_bssr1.py --large <dir>/bssr1 ../data/bssr1_set2_face.npz ../data/bssr1_set3_finger.npz
   ```

   which writes the three `.npz` files read by `exp_bssr1.py`.
