import json
from pathlib import Path
from health_ai.core.exceptions import StorageError


class JSONStore:

    def save(self, path: Path, data: dict):
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4, default=str)
        except Exception as e:
            raise StorageError(str(e))

    def load(self, path: Path):
        try:
            if not path.exists():
                return None
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise StorageError(str(e))
