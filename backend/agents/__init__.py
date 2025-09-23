# agents/__init__.py
from .base import BaseAgent
from .llm_localmodel import LocalAgent
from .llm_openrouter import OpenrouterAgent
from .preprocessing import PreprocessingAgent

__all__ = ['BaseAgent', 'LocalAgent', 'OpenrouterAgent', 'PreprocessingAgent']