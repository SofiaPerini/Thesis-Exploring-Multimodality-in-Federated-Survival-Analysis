import os


BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", "/data/zoorvival/TCGA")

AVAILABLE_DATASETS = [
    "BLCA",
    "BRCA",
    "LUAD",
    "UCEC",
]
