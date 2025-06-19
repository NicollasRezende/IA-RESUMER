"""
Teste simples do sistema de transcrição.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_import():
    """Testa se os módulos podem ser importados."""
    try:
        from src.core.transcriber import WhisperTranscriber
        from src.core.audio_processor import AudioProcessor
        from src.core.config import WHISPER_CONFIG
        print("✅ Todos os módulos importados com sucesso!")
        return True
    except ImportError as e:
        print(f"❌ Erro ao importar módulos: {e}")
        return False

if __name__ == "__main__":
    test_import()
