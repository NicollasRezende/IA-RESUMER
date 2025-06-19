"""
Módulo para processamento e manipulação de arquivos de áudio.
"""
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

import ffmpeg
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from rich.console import Console
from rich.progress import track

from .config import (
    AUDIO_CONFIG,
    SUPPORTED_AUDIO_FORMATS,
    TEMP_DIR
)

console = Console()


class AudioProcessor:
    """Classe para processamento de arquivos de áudio."""
    
    def __init__(self):
        """Inicializa o processador de áudio."""
        self.sample_rate = AUDIO_CONFIG["sample_rate"]
        self.channels = AUDIO_CONFIG["channels"]
        self.chunk_duration = AUDIO_CONFIG["chunk_duration"]
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Verifica se o FFmpeg está instalado."""
        try:
            subprocess.run(["ffmpeg", "-version"], 
                         capture_output=True, 
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("[bold red]⚠️  FFmpeg não encontrado![/bold red]")
            console.print("Instale o FFmpeg: https://ffmpeg.org/download.html")
            raise RuntimeError("FFmpeg é necessário para processar áudio")
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """
        Verifica se o formato do arquivo é suportado.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            True se o formato é suportado
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS
    
    def extract_audio_from_video(
        self, 
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Extrai áudio de um arquivo de vídeo.
        
        Args:
            video_path: Caminho do vídeo
            output_path: Caminho de saída (opcional)
            
        Returns:
            Path do arquivo de áudio extraído
        """
        video_path = Path(video_path)
        if output_path is None:
            output_path = TEMP_DIR / f"{video_path.stem}_audio.wav"
        else:
            output_path = Path(output_path)
        
        console.print(f"🎬 Extraindo áudio de: [bold]{video_path.name}[/bold]")
        
        try:
            # Usar ffmpeg-python para extrair áudio
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream, 
                str(output_path),
                acodec='pcm_s16le',  # WAV format
                ac=self.channels,     # Mono
                ar=self.sample_rate   # 16kHz
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            console.print(f"✅ Áudio extraído: [bold green]{output_path.name}[/bold green]")
            return output_path
            
        except ffmpeg.Error as e:
            console.print(f"[bold red]❌ Erro ao extrair áudio: {e}[/bold red]")
            raise
    
    def convert_to_wav(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Converte qualquer formato de áudio para WAV.
        
        Args:
            audio_path: Caminho do áudio original
            output_path: Caminho de saída (opcional)
            
        Returns:
            Path do arquivo WAV
        """
        audio_path = Path(audio_path)
        
        # Se já é WAV com as especificações corretas, retornar
        if audio_path.suffix.lower() == '.wav':
            audio = AudioSegment.from_wav(str(audio_path))
            if (audio.frame_rate == self.sample_rate and 
                audio.channels == self.channels):
                return audio_path
        
        if output_path is None:
            output_path = TEMP_DIR / f"{audio_path.stem}_converted.wav"
        else:
            output_path = Path(output_path)
        
        console.print(f"🔄 Convertendo áudio: [bold]{audio_path.name}[/bold]")
        
        # Carregar áudio
        audio = AudioSegment.from_file(str(audio_path))
        
        # Converter para mono e 16kHz
        audio = audio.set_channels(self.channels)
        audio = audio.set_frame_rate(self.sample_rate)
        
        # Exportar como WAV
        audio.export(
            str(output_path),
            format="wav",
            parameters=["-acodec", "pcm_s16le"]
        )
        
        console.print(f"✅ Áudio convertido: [bold green]{output_path.name}[/bold green]")
        return output_path
    
    def split_audio_by_duration(
        self,
        audio_path: Union[str, Path],
        chunk_duration: Optional[int] = None
    ) -> List[Path]:
        """
        Divide o áudio em chunks de duração fixa.
        
        Args:
            audio_path: Caminho do áudio
            chunk_duration: Duração de cada chunk em segundos
            
        Returns:
            Lista de caminhos dos chunks
        """
        audio_path = Path(audio_path)
        chunk_duration = chunk_duration or self.chunk_duration
        
        audio = AudioSegment.from_file(str(audio_path))
        duration_ms = len(audio)
        chunk_duration_ms = chunk_duration * 1000
        
        chunks = []
        chunk_paths = []
        
        # Dividir em chunks
        for i in range(0, duration_ms, chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            chunks.append(chunk)
        
        console.print(f"✂️  Dividindo áudio em {len(chunks)} partes de {chunk_duration}s")
        
        # Salvar chunks
        for i, chunk in enumerate(track(chunks, description="Salvando chunks...")):
            chunk_path = TEMP_DIR / f"{audio_path.stem}_chunk_{i:03d}.wav"
            chunk.export(str(chunk_path), format="wav")
            chunk_paths.append(chunk_path)
        
        return chunk_paths
    
    def split_audio_by_silence(
        self,
        audio_path: Union[str, Path],
        min_silence_len: int = 1000,
        silence_thresh: Optional[int] = None,
        keep_silence: int = 500
    ) -> List[Path]:
        """
        Divide o áudio nos momentos de silêncio.
        
        Args:
            audio_path: Caminho do áudio
            min_silence_len: Duração mínima de silêncio (ms)
            silence_thresh: Threshold de silêncio (dB)
            keep_silence: Quantidade de silêncio para manter (ms)
            
        Returns:
            Lista de caminhos dos chunks
        """
        audio_path = Path(audio_path)
        silence_thresh = silence_thresh or AUDIO_CONFIG["silence_threshold"]
        
        console.print(f"🔇 Detectando silêncios em: [bold]{audio_path.name}[/bold]")
        
        audio = AudioSegment.from_file(str(audio_path))
        
        # Dividir por silêncio
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        
        if not chunks:
            console.print("[yellow]⚠️  Nenhum silêncio detectado, usando divisão por duração[/yellow]")
            return self.split_audio_by_duration(audio_path)
        
        console.print(f"✂️  Áudio dividido em {len(chunks)} segmentos")
        
        # Salvar chunks
        chunk_paths = []
        for i, chunk in enumerate(track(chunks, description="Salvando segmentos...")):
            chunk_path = TEMP_DIR / f"{audio_path.stem}_segment_{i:03d}.wav"
            chunk.export(str(chunk_path), format="wav")
            chunk_paths.append(chunk_path)
        
        return chunk_paths
    
    def merge_audio_files(
        self,
        audio_paths: List[Union[str, Path]],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Mescla múltiplos arquivos de áudio em um único arquivo.
        
        Args:
            audio_paths: Lista de caminhos de áudio
            output_path: Caminho de saída
            
        Returns:
            Path do arquivo mesclado
        """
        output_path = Path(output_path)
        
        console.print(f"🔀 Mesclando {len(audio_paths)} arquivos de áudio")
        
        # Carregar primeiro áudio
        combined = AudioSegment.from_file(str(audio_paths[0]))
        
        # Adicionar os demais
        for path in track(audio_paths[1:], description="Mesclando arquivos..."):
            audio = AudioSegment.from_file(str(path))
            combined += audio
        
        # Exportar
        combined.export(str(output_path), format="wav")
        
        console.print(f"✅ Áudio mesclado: [bold green]{output_path.name}[/bold green]")
        return output_path
    
    def get_audio_info(self, audio_path: Union[str, Path]) -> dict:
        """
        Obtém informações sobre o arquivo de áudio.
        
        Args:
            audio_path: Caminho do áudio
            
        Returns:
            Dict com informações do áudio
        """
        audio_path = Path(audio_path)
        
        try:
            probe = ffmpeg.probe(str(audio_path))
            audio_stream = next(
                (stream for stream in probe['streams'] 
                 if stream['codec_type'] == 'audio'), 
                None
            )
            
            if audio_stream:
                duration = float(probe['format']['duration'])
                return {
                    "filename": audio_path.name,
                    "format": probe['format']['format_name'],
                    "duration": duration,
                    "duration_formatted": self._format_duration(duration),
                    "codec": audio_stream['codec_name'],
                    "sample_rate": int(audio_stream['sample_rate']),
                    "channels": audio_stream['channels'],
                    "bitrate": int(probe['format']['bit_rate']) // 1000,  # kbps
                    "size": os.path.getsize(audio_path) / (1024 * 1024),  # MB
                }
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Erro ao obter informações: {e}[/yellow]")
        
        # Fallback usando pydub
        audio = AudioSegment.from_file(str(audio_path))
        return {
            "filename": audio_path.name,
            "duration": len(audio) / 1000,  # segundos
            "duration_formatted": self._format_duration(len(audio) / 1000),
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "size": os.path.getsize(audio_path) / (1024 * 1024),  # MB
        }
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Formata duração em segundos para formato legível."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def normalize_audio(
        self,
        audio_path: Union[str, Path],
        target_dBFS: float = -20.0
    ) -> Path:
        """
        Normaliza o volume do áudio.
        
        Args:
            audio_path: Caminho do áudio
            target_dBFS: Nível alvo em dBFS
            
        Returns:
            Path do áudio normalizado
        """
        audio_path = Path(audio_path)
        output_path = TEMP_DIR / f"{audio_path.stem}_normalized.wav"
        
        audio = AudioSegment.from_file(str(audio_path))
        
        # Calcular diferença para o alvo
        change_in_dBFS = target_dBFS - audio.dBFS
        
        # Normalizar
        normalized = audio.apply_gain(change_in_dBFS)
        
        # Exportar
        normalized.export(str(output_path), format="wav")
        
        console.print(f"🔊 Áudio normalizado para {target_dBFS} dBFS")
        return output_path
    
    def clean_temp_files(self):
        """Remove arquivos temporários."""
        temp_files = list(TEMP_DIR.glob("*"))
        if temp_files:
            console.print(f"🧹 Limpando {len(temp_files)} arquivos temporários")
            for file in temp_files:
                try:
                    file.unlink()
                except Exception as e:
                    console.print(f"[yellow]⚠️  Erro ao remover {file.name}: {e}[/yellow]")