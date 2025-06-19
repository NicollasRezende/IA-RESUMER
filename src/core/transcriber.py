"""
Módulo principal de transcrição usando OpenAI Whisper.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

import whisper
import torch
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .config import (
    WHISPER_CONFIG, 
    TRANSCRIPT_DIR, 
    get_model_info,
    OUTPUT_CONFIG
)

# Suprimir avisos do Whisper
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()


class WhisperTranscriber:
    """Classe principal para transcrição de áudio usando Whisper."""
    
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None):
        """
        Inicializa o transcritor.
        
        Args:
            model_size: Tamanho do modelo (tiny, base, small, medium, large)
            device: Dispositivo para execução (cpu, cuda)
        """
        self.model_size = model_size or WHISPER_CONFIG["model_size"]
        self.device = device or WHISPER_CONFIG["device"]
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Carrega o modelo Whisper com feedback visual."""
        model_info = get_model_info(self.model_size)
        
        console.print(f"\n🎙️  Carregando modelo Whisper [bold cyan]{self.model_size}[/bold cyan]")
        console.print(f"   Tamanho: {model_info['size']} | VRAM necessária: {model_info['vram']}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Baixando e carregando modelo...", total=None)
            
            start_time = time.time()
            self.model = whisper.load_model(
                self.model_size, 
                device=self.device,
                download_root=None  # Usa o diretório padrão ~/.cache/whisper
            )
            
            elapsed = time.time() - start_time
            progress.update(task, completed=True)
            
        console.print(f"✅ Modelo carregado em {elapsed:.1f}s no dispositivo: [bold green]{self.device}[/bold green]\n")
    
    def transcribe(
        self, 
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict:
        """
        Transcreve um arquivo de áudio.
        
        Args:
            audio_path: Caminho para o arquivo de áudio
            language: Idioma do áudio (None para detecção automática)
            task: 'transcribe' ou 'translate'
            initial_prompt: Prompt inicial para guiar o modelo
            temperature: Temperatura para geração (0 = determinístico)
            **kwargs: Argumentos adicionais para whisper.transcribe
            
        Returns:
            Dict com transcrição e metadados
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        
        console.print(f"🎵 Processando: [bold]{audio_path.name}[/bold]")
        
        # Configurar parâmetros
        params = {
            "language": language or WHISPER_CONFIG["language"],
            "task": task,
            "fp16": WHISPER_CONFIG["fp16"] and self.device == "cuda",
            "verbose": WHISPER_CONFIG["verbose"],
            "temperature": temperature,
            "word_timestamps": OUTPUT_CONFIG["include_timestamps"],
        }
        
        if initial_prompt:
            params["initial_prompt"] = initial_prompt
        
        # Atualizar com kwargs adicionais
        params.update(kwargs)
        
        # Transcrever com progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Transcrevendo áudio...", total=100)
            
            start_time = time.time()
            
            # Callback simulado (Whisper não oferece progress real)
            def progress_callback(seek, total):
                progress.update(task, completed=int((seek / total) * 100))
            
            result = self.model.transcribe(str(audio_path), **params)
            
            progress.update(task, completed=100)
            elapsed = time.time() - start_time
        
        # Adicionar metadados
        # Calcular duração do áudio se não fornecida pelo Whisper
        audio_duration = result.get("duration", 0)
        if audio_duration == 0 and "segments" in result and result["segments"]:
            # Pegar o timestamp final do último segmento
            audio_duration = result["segments"][-1]["end"]
        
        result["metadata"] = {
            "file": audio_path.name,
            "duration": audio_duration,
            "processing_time": elapsed,
            "model": self.model_size,
            "language": result.get("language", params["language"]),
            "device": self.device,
        }
        
        # Calcular velocidade de processamento
        if result["metadata"]["duration"] > 0:
            speed_ratio = result["metadata"]["duration"] / elapsed
            console.print(
                f"✅ Transcrição concluída em {elapsed:.1f}s "
                f"([bold green]{speed_ratio:.1f}x[/bold green] tempo real)"
            )
        
        return result
    
    def transcribe_with_segments(
        self, 
        audio_path: Union[str, Path],
        **kwargs
    ) -> Dict:
        """
        Transcreve áudio retornando segmentos detalhados com timestamps.
        
        Args:
            audio_path: Caminho para o arquivo de áudio
            **kwargs: Argumentos para transcribe()
            
        Returns:
            Dict com transcrição completa e segmentos
        """
        result = self.transcribe(audio_path, **kwargs)
        
        # Formatar segmentos
        segments = []
        for segment in result.get("segments", []):
            seg_data = {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "duration": segment["end"] - segment["start"],
            }
            
            # Adicionar palavras com timestamps se disponível
            if "words" in segment and OUTPUT_CONFIG["include_timestamps"]:
                seg_data["words"] = [
                    {
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word.get("probability", 1.0)
                    }
                    for word in segment["words"]
                ]
            
            segments.append(seg_data)
        
        return {
            "text": result["text"],
            "segments": segments,
            "metadata": result["metadata"]
        }
    
    def save_transcription(
        self, 
        result: Dict,
        output_path: Optional[Path] = None,
        format: str = "json"
    ) -> Path:
        """
        Salva a transcrição em arquivo.
        
        Args:
            result: Resultado da transcrição
            output_path: Caminho de saída (None para usar padrão)
            format: Formato de saída (json, txt, srt, vtt)
            
        Returns:
            Path do arquivo salvo
        """
        if output_path is None:
            filename = Path(result["metadata"]["file"]).stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = TRANSCRIPT_DIR / f"{filename}_{timestamp}.{format}"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        elif format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
        
        elif format == "srt":
            content = self._to_srt(result.get("segments", []))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        elif format == "vtt":
            content = self._to_vtt(result.get("segments", []))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        else:
            raise ValueError(f"Formato não suportado: {format}")
        
        console.print(f"💾 Transcrição salva em: [bold]{output_path}[/bold]")
        return output_path
    
    def _to_srt(self, segments: List[Dict]) -> str:
        """Converte segmentos para formato SRT."""
        lines = []
        for i, segment in enumerate(segments, 1):
            start = self._seconds_to_time(segment["start"])
            end = self._seconds_to_time(segment["end"])
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(segment["text"])
            lines.append("")
        return "\n".join(lines)
    
    def _to_vtt(self, segments: List[Dict]) -> str:
        """Converte segmentos para formato WebVTT."""
        lines = ["WEBVTT", ""]
        for segment in segments:
            start = self._seconds_to_time(segment["start"], vtt=True)
            end = self._seconds_to_time(segment["end"], vtt=True)
            lines.append(f"{start} --> {end}")
            lines.append(segment["text"])
            lines.append("")
        return "\n".join(lines)
    
    @staticmethod
    def _seconds_to_time(seconds: float, vtt: bool = False) -> str:
        """Converte segundos para formato de tempo."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        else:
            # SRT usa vírgula como separador decimal
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
    
    def detect_language(self, audio_path: Union[str, Path]) -> str:
        """
        Detecta o idioma do áudio.
        
        Args:
            audio_path: Caminho para o arquivo de áudio
            
        Returns:
            Código do idioma detectado
        """
        audio_path = Path(audio_path)
        
        console.print(f"🔍 Detectando idioma de: [bold]{audio_path.name}[/bold]")
        
        # Detectar usando os primeiros 30 segundos
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)
        
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        
        console.print(f"🌐 Idioma detectado: [bold green]{detected_lang}[/bold green] "
                     f"(confiança: {probs[detected_lang]:.1%})")
        
        return detected_lang
    
    def get_device_info(self) -> Dict:
        """Retorna informações sobre o dispositivo em uso."""
        info = {
            "device": self.device,
            "device_name": torch.cuda.get_device_name() if self.device == "cuda" else "CPU",
            "cuda_available": torch.cuda.is_available(),
        }
        
        if self.device == "cuda":
            info.update({
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
            })
        
        return info