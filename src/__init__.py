# src/__init__.py
"""Sistema de Transcrição de Áudio e Vídeo."""
__version__ = "1.0.0"

# src/core/__init__.py
"""Módulos principais do sistema."""
from src.core.transcriber import WhisperTranscriber
from src.core.audio_processor import AudioProcessor

__all__ = ["WhisperTranscriber", "AudioProcessor"]

# src/utils/__init__.py
"""Utilitários do sistema."""

# src/api/__init__.py
"""API do sistema (futuro)."""

# tests/__init__.py
"""Testes do sistema."""