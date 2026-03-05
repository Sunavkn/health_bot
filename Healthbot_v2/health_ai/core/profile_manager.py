from pathlib import Path
from health_ai.config.settings import BASE_DIR


class ProfileManager:

    @staticmethod
    def get_profile_dir(profile_id: str) -> Path:
        profile_dir = BASE_DIR / "data" / "profiles" / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir

    @staticmethod
    def get_vector_dir(profile_id: str) -> Path:
        vector_dir = ProfileManager.get_profile_dir(profile_id) / "vector_store"
        vector_dir.mkdir(parents=True, exist_ok=True)
        return vector_dir

    @staticmethod
    def get_static_dir(profile_id: str) -> Path:
        static_dir = ProfileManager.get_profile_dir(profile_id) / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        return static_dir

    @staticmethod
    def get_dynamic_dir(profile_id: str) -> Path:
        dynamic_dir = ProfileManager.get_profile_dir(profile_id) / "dynamic"
        dynamic_dir.mkdir(parents=True, exist_ok=True)
        return dynamic_dir
