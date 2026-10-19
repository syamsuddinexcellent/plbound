"""Parse NIST BSSR1 Set 1 (fing_x_face) into four 517 x 517 similarity matrices.

Usage:  python prepare_bssr1.py <path-to-extracted>/bssr1/fing_x_face  ../data/bssr1_set1.npz
        python prepare_bssr1.py --large <path-to-extracted>/bssr1 ../data/bssr1_set2_face.npz ../data/bssr1_set3_finger.npz

Rows follow users.xml (probe samples), columns follow enrollees.xml (gallery).
genuine[i, j] is True when probe i and gallery j share the same subject_id.
Data source: https://www.nist.gov/itl/iad/btg/nist-biometric-scores-set-bssr1
"""
import re
import sys
from pathlib import Path

import numpy as np

MATCHERS = {"face_C": "face/C", "face_G": "face/G", "finger_LI": "li/V", "finger_RI": "ri/V"}


def read_signature_set(path):
    pat = re.compile(r'<signature name="([^"]+)" subject_id="([^"]+)"/>')
    return pat.findall(Path(path).read_text())


def read_similarity_file(path):
    lines = Path(path).read_text().splitlines()  # header line, count, then one score per line
    n = int(lines[1])
    scores = np.array(lines[2:2 + n], dtype=float)
    assert scores.size == n
    return scores


def main(root, out):
    root = Path(root)
    users = read_signature_set(root / "sets/dos/users.xml")
    enrollees = read_signature_set(root / "sets/dos/enrollees.xml")
    uid = np.array([s for _, s in users])
    eid = np.array([s for _, s in enrollees])
    genuine = uid[:, None] == eid[None, :]
    mats = {}
    for key, sub in MATCHERS.items():
        mats[key] = np.vstack([read_similarity_file(root / "sims/dos" / sub / name) for name, _ in users])
    np.savez_compressed(out, genuine=genuine, user_id=uid, enrollee_id=eid, **mats)
    print(f"users={len(users)} enrollees={len(enrollees)} genuine pairs={genuine.sum()}")
    for key, m in mats.items():
        print(f"{key}: shape={m.shape} min={m.min():.4g} max={m.max():.4g} "
              f"genuine mean={m[genuine].mean():.4g} impostor mean={m[~genuine].mean():.4g}")


if __name__ == "__main__" and len(sys.argv) == 3:
    main(sys.argv[1], sys.argv[2])


# ---------------------------------------------------------------------------------------------
# Sets 2 (face_x_face, 3000 subjects) and 3 (fing_x_fing, 6000 subjects)
# ---------------------------------------------------------------------------------------------
def read_fast(path):
    with open(path) as fh:
        fh.readline()
        n = int(fh.readline())
        v = np.array(fh.read().split()[:n], dtype=np.float32)   # files end with a footer line
    assert v.size == n
    return v


def prepare_large(root, out_face, out_finger):
    root = Path(root)
    # Set 2: two probes per subject, matchers C and G
    users = read_signature_set(root / "face_x_face/sets/dos/users.xml")
    enr = read_signature_set(root / "face_x_face/sets/dos/enrollees.xml")
    mats = {k: np.vstack([read_fast(root / "face_x_face/sims/dos/face" / k / n) for n, _ in users])
            for k in ("C", "G")}
    np.savez_compressed(out_face, face_C=mats["C"], face_G=mats["G"],
                        probe_id=np.array([s for _, s in users]), enrollee_id=np.array([s for _, s in enr]))
    print("set 2:", mats["C"].shape, "genuine:", (np.array([s for _, s in users])[:, None] ==
                                                  np.array([s for _, s in enr])[None, :]).sum())
    # Set 3: left and right index fingers, aligned by subject
    ids = {}
    mats = {}
    for f in ("li", "ri"):
        users = read_signature_set(root / f"fing_x_fing/sets/dos/{f}/users.xml")
        enr = read_signature_set(root / f"fing_x_fing/sets/dos/{f}/enrollees.xml")
        ids[f] = (np.array([s for _, s in users]), np.array([s for _, s in enr]))
        mats[f] = np.vstack([read_fast(root / f"fing_x_fing/sims/dos/{f}/V" / n) for n, _ in users])
    pu = np.argsort(ids["ri"][0])[np.argsort(np.argsort(ids["li"][0]))]   # align ri rows to li order
    pe = np.argsort(ids["ri"][1])[np.argsort(np.argsort(ids["li"][1]))]
    ri = mats["ri"][pu][:, pe]
    assert np.all(ids["ri"][0][pu] == ids["li"][0]) and np.all(ids["ri"][1][pe] == ids["li"][1])
    np.savez_compressed(out_finger, finger_LI=mats["li"], finger_RI=ri,
                        probe_id=ids["li"][0], enrollee_id=ids["li"][1])
    print("set 3:", mats["li"].shape)


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--large":
    prepare_large(sys.argv[2], sys.argv[3], sys.argv[4])
