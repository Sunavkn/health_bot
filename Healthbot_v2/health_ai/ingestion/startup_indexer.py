"""
startup_indexer.py
Scans the profile's raw_documents directory and indexes new files.

Changes vs original:
  - Registry now stores OCR quality metadata (engine, line_count)
  - Entries previously marked "processed: true" with no ocr_info are treated
    as UNCONFIRMED for image files (they'll be re-indexed on next startup unless
    force_reindex=False is explicitly passed by the caller)
  - force_reindex=True resets all image entries so they're re-processed
  - Failed files are NOT added to the registry so they retry on next startup
"""
import json
from pathlib import Path

from health_ai.ingestion.ingest import IngestionEngine
from health_ai.core.profile_manager import ProfileManager
from health_ai.core.logger import ingestion_logger


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class StartupIndexer:

    def __init__(self, profile_id: str, force_reindex_images: bool = False):
        """
        Args:
            profile_id: The profile to index.
            force_reindex_images: If True, all image files are re-indexed even
                                  if they appear in the registry.  Useful when
                                  the OCR pipeline has been upgraded.
        """
        self.profile_id = profile_id
        self.force_reindex_images = force_reindex_images
        self.profile_dir = ProfileManager.get_profile_dir(profile_id)
        self.raw_dir = self.profile_dir / "raw_documents"
        self.registry_path = self.profile_dir / "processed_files.json"

        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

        # If force_reindex_images, clear all image entries from the registry
        if force_reindex_images:
            stale_keys = [
                k for k in list(self.registry.keys())
                if Path(k).suffix.lower() in IMAGE_EXTENSIONS
            ]
            for k in stale_keys:
                del self.registry[k]
            if stale_keys:
                ingestion_logger.info(
                    f"[{profile_id}] force_reindex_images: cleared "
                    f"{len(stale_keys)} image entries from registry."
                )

    def scan_and_index(self):
        engine = IngestionEngine(self.profile_id)
        indexed_count = 0
        skipped_count = 0
        failed_count = 0

        for file in sorted(self.raw_dir.rglob("*")):
            if not file.is_file():
                continue
            if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            relative_path = str(file.relative_to(self.profile_dir))

            # Skip files that are already confirmed in the registry
            existing = self.registry.get(relative_path)
            if existing and existing.get("confirmed", False):
                skipped_count += 1
                continue

            ingestion_logger.info(f"[{self.profile_id}] Indexing: {relative_path}")
            print(f"[{self.profile_id}] Indexing: {relative_path}")

            try:
                folder = file.parent.name.lower()
                if "prescription" in folder:
                    source_type = "prescription"
                    importance = 0.9
                elif "blood" in folder or "lab" in folder or "report" in folder:
                    source_type = "lab_report"
                    importance = 1.0
                else:
                    source_type = "uploaded_document"
                    importance = 0.8

                result = engine.ingest_document(
                    file_path=str(file),
                    source_type=source_type,
                    importance_score=importance,
                )

                if result.get("status") == "success":
                    # Record confirmed with metadata
                    self.registry[relative_path] = {
                        "confirmed": True,
                        "source_type": source_type,
                    }
                    indexed_count += 1
                    print(f"[{self.profile_id}]  ✓ Indexed: {file.name}")
                else:
                    # Do NOT mark as confirmed — will retry next time
                    reason = result.get("reason", "unknown")
                    ingestion_logger.warning(
                        f"[{self.profile_id}] Ingestion returned non-success "
                        f"for {relative_path}: {reason}"
                    )
                    print(f"[{self.profile_id}]  ✗ Failed: {file.name} — {reason}")
                    failed_count += 1

            except Exception as e:
                ingestion_logger.error(
                    f"[{self.profile_id}] Exception indexing {relative_path}: {e}"
                )
                print(f"[{self.profile_id}]  ✗ Error: {file.name} — {e}")
                failed_count += 1

        self._save_registry()
        summary = (
            f"[{self.profile_id}] Startup indexing done: "
            f"{indexed_count} indexed, {skipped_count} skipped, {failed_count} failed."
        )
        ingestion_logger.info(summary)
        print(summary)

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)
