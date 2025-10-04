from typing import List
from pathlib import Path

from models.configs import DatasetConfig
from models import Document


class DataLoader:
    def __init__(self, config: DatasetConfig):
        if not isinstance(config, DatasetConfig):
            raise TypeError("DataLoader expects a DatasetConfig instance")
        self.config: DatasetConfig = config

    def load(self) -> List[Document]:
        base_path = Path(self.config.path)
        if not base_path.exists():
            raise FileNotFoundError(f"Invalid path: {base_path}")

        allowed = {ext.lower().lstrip(".") for ext in self.config.allowed_types}
        documents: List[Document] = []

        if base_path.is_file():
            # Handle single file
            if base_path.suffix.lower().lstrip(".") in allowed:
                text = base_path.read_text(encoding="utf-8", errors="ignore")
                documents.append(
                    Document(
                        name=base_path.name,
                        path=str(base_path),
                        text=text
                    )
                )
        elif base_path.is_dir():
            # Handle directory
            for file in base_path.rglob("*"):  # recursive walk
                if file.is_file() and file.suffix.lower().lstrip(".") in allowed:
                    text = file.read_text(encoding="utf-8", errors="ignore")
                    documents.append(
                        Document(
                            name=file.name,
                            path=str(file),
                            text=text
                        )
                    )
        else:
            raise FileNotFoundError(f"Path is neither a file nor a directory: {base_path}")

        return documents

