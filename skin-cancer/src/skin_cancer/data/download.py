"""Fetch the raw datasets into SKIN_DATA_DIR (default ~/data/skin-cancer).

Raw data is licensed (CC-BY-NC / research-only) and is never committed to the repo.
Usage: uv run python -m skin_cancer.data.download [isic2024 ham10000 isic2018_task2 ph2]
"""
import sys
import urllib.request

from skin_cancer.config import DATA_DIR

S3 = "https://isic-challenge-data.s3.amazonaws.com"
DV = "https://dataverse.harvard.edu/api/access/datafile"

SOURCES = {
    # ISIC 2024 SLICE-3D, CC-BY-NC 4.0, doi:10.34970/2024-slice-3d
    "isic2024": {
        "ISIC_2024_Training_GroundTruth.csv": f"{S3}/2024/ISIC_2024_Training_GroundTruth.csv",
        "ISIC_2024_Training_Supplement.csv": f"{S3}/2024/ISIC_2024_Training_Supplement.csv",
        "ISIC_2024_Training_Input.zip": f"{S3}/2024/ISIC_2024_Training_Input.zip",
    },
    # HAM10000, CC-BY-NC 4.0, doi:10.7910/DVN/DBW86T
    "ham10000": {
        "HAM10000_metadata.tab": f"{DV}/4338392?format=original",
        "HAM10000_segmentations_lesion_tschandl.zip": f"{DV}/3838943",
        "HAM10000_images_part_1.zip": f"{DV}/3172585",
        "HAM10000_images_part_2.zip": f"{DV}/3172584",
    },
    # ISIC 2018 Task 2 attribute masks, CC-BY-NC
    "isic2018_task2": {
        "ISIC2018_Task2_Training_GroundTruth_v3.zip":
            f"{S3}/2018/ISIC2018_Task2_Training_GroundTruth_v3.zip",
    },
    # PH2 (Mendonca 2013). Official link is dead; these are UNOFFICIAL mirrors.
    # Terms: research/education only, no redistribution. Keep local.
    "ph2": {
        "PH2.zip": "https://zenodo.org/api/records/17498821/files/PH2.zip/content",
        "PH2_dataset.txt":
            "https://raw.githubusercontent.com/vikaschouhan/PH2-dataset/master/PH2_dataset.txt",
    },
}


def fetch(name: str) -> None:
    dest = DATA_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    for fname, url in SOURCES[name].items():
        out = dest / fname
        if out.exists():
            print(f"skip {out} (exists)")
            continue
        print(f"get  {url} -> {out}")
        urllib.request.urlretrieve(url, out)


if __name__ == "__main__":
    for n in sys.argv[1:] or SOURCES:
        fetch(n)
