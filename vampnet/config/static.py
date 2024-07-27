import vampnet

RUNS_DIR = vampnet.ROOT / "runs" / vampnet.CONFIG 
DB = str(vampnet.ROOT / "data" / "sqlit3" / vampnet.DB_FILE)
CACHE_PATH = vampnet.ROOT / "data" / "cache" / vampnet.DB_FILE

REPO_ID = f"{vampnet.HF_USERNAME}/{vampnet.HF_REPO_NAME}"
MODEL_FILE = vampnet.MODELS_DIR / f"{vampnet.CONFIG}-{vampnet.EXPORT_MODEL_TAG}.vampnet"