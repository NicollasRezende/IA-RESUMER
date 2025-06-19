"""
Configurações centralizadas do sistema de transcrição.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Diretórios base
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
TEMP_DIR = DATA_DIR / "temp"

# Criar diretórios se não existirem
for directory in [UPLOAD_DIR, TRANSCRIPT_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configurações do Whisper
WHISPER_CONFIG = {
    "model_size": os.getenv("WHISPER_MODEL", "small"),  # tiny, base, small, medium, large
    "device": os.getenv("WHISPER_DEVICE", "cpu"),       # cpu, cuda
    "language": os.getenv("WHISPER_LANGUAGE", "pt"),    # português por padrão
    "task": "transcribe",                                # transcribe ou translate
    "fp16": os.getenv("WHISPER_FP16", "False").lower() == "true",
    "verbose": os.getenv("WHISPER_VERBOSE", "False").lower() == "true",
}

# Formatos de áudio suportados
SUPPORTED_AUDIO_FORMATS = [
    ".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg",
    ".wma", ".aac", ".opus", ".webm", ".mkv", ".avi"
]

# Configurações de processamento de áudio
AUDIO_CONFIG = {
    "sample_rate": 16000,           # Taxa de amostragem para Whisper
    "channels": 1,                  # Mono
    "chunk_duration": 30,           # Segundos por chunk para arquivos longos
    "max_file_size": 2 * 1024**3,   # 2GB máximo
    "silence_threshold": -40,       # dB para detecção de silêncio
    "min_silence_duration": 1000,   # ms de silêncio para split
}

# Configurações de performance
PERFORMANCE_CONFIG = {
    "batch_size": int(os.getenv("BATCH_SIZE", "1")),
    "num_workers": int(os.getenv("NUM_WORKERS", "4")),
    "cache_enabled": os.getenv("CACHE_ENABLED", "True").lower() == "true",
    "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),  # segundos
}

# Configurações de logging
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "transcription.log",
}

# Configurações de output
OUTPUT_CONFIG = {
    "include_timestamps": True,
    "include_confidence": True,
    "format": "json",  # json, txt, srt, vtt
    "save_audio_chunks": False,
}

def get_model_info(model_size: str) -> Dict[str, any]:
    """Retorna informações sobre o modelo Whisper selecionado."""
    model_info = {
        "tiny": {"size": "39M", "vram": "~1GB", "relative_speed": 32},
        "base": {"size": "74M", "vram": "~1GB", "relative_speed": 16},
        "small": {"size": "244M", "vram": "~2GB", "relative_speed": 8},
        "medium": {"size": "769M", "vram": "~5GB", "relative_speed": 4},
        "large": {"size": "1550M", "vram": "~10GB", "relative_speed": 1},
    }
    return model_info.get(model_size, model_info["small"])

def validate_config() -> bool:
    """Valida as configurações do sistema."""
    import torch
    
    # Verificar se o dispositivo está disponível
    if WHISPER_CONFIG["device"] == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA solicitado mas não disponível. Usando CPU.")
        WHISPER_CONFIG["device"] = "cpu"
    
    # Verificar se o modelo é válido
    valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    if WHISPER_CONFIG["model_size"] not in valid_models:
        print(f"⚠️  Modelo inválido: {WHISPER_CONFIG['model_size']}. Usando 'small'.")
        WHISPER_CONFIG["model_size"] = "small"
    
    return True

# Validar configurações ao importar
validate_config()