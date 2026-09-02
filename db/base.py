from abc import ABC, abstractmethod
from typing import List, Optional

from models import ProjectData


class StorageBackend(ABC):
    @abstractmethod
    def load(self, rera_id: str) -> Optional[ProjectData]: ...

    @abstractmethod
    def save(self, project: ProjectData) -> None: ...

    def get_document_urls(self, rera_id: str) -> Optional[List[dict]]:
        """
        Return a list of {"url": str, "category": str} dicts for the given rera_id,
        or None if this backend does not support URL auto-fetching (default).
        """
        return None
