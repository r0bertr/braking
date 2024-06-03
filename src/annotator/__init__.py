from pathlib import Path

from braking import DATA_ROOT

PATH_TO_TEMP = Path("/home/user/temp")
PATH_TO_SESSIONS = DATA_ROOT / "databases/cleaned/annotation_sessions"
PATH_TO_DB = DATA_ROOT / "databases/cleaned/probe_data.db"
CLIP_LENGTH = 100

if not PATH_TO_TEMP.exists():
    PATH_TO_TEMP.mkdir(parents=True)
