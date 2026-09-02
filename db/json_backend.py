import json
import os
from typing import Optional

from filelock import FileLock

import config
from db.base import StorageBackend
from models import ProjectData


class JSONStorageBackend(StorageBackend):
    def __init__(self):
        os.makedirs(config.STORAGE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)

    def load(self, rera_id: str) -> Optional[ProjectData]:
        path = os.path.join(config.STORAGE_DIR, f"{rera_id.replace('/', '_')}.json")
        with FileLock(path + ".lock"):
            if not os.path.exists(path):
                return None
            with open(path, "r") as f:
                return ProjectData(**json.load(f))

    def save(self, project: ProjectData) -> None:
        path = os.path.join(config.STORAGE_DIR, f"{project.rera_id.replace('/', '_')}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with FileLock(path + ".lock"):
            with open(path, "w") as f:
                f.write(project.model_dump_json(indent=2))

