from __future__ import annotations
from dotenv import load_dotenv
from ..config import settings  # re-export

load_dotenv()

__all__ = ["settings"]

