"""
Validadores e verificações para o sistema de transcrição.
"""
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import mimetypes
import magic
from urllib.parse import urlparse
import re

from rich.console import Console
from rich.table import Table

from ..core.config import (
    SUPPORTED_AUDIO_FORMATS,
    AUDIO_CONFIG,
    WHISPER_CONFIG
)

console = Console()


class MediaValidator:
    """Validador de arquivos de mídia."""
    
    def __init__(self):
        """Inicializa o validador."""
        self.supported_formats = SUPPORTED_AUDIO_FORMATS
        self.max_file_size = AUDIO_CONFIG["max_file_size"]
        self._init_magic()
    
    def _init_magic(self):
        """Inicializa a biblioteca magic para detecção de tipos."""
        try:
            self.magic = magic.Magic(mime=True)
        except Exception:
            console.print("[yellow]⚠️  python-magic não disponível, usando detecção básica[/yellow]")
            self.magic = None
    
    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """
        Valida um arquivo de mídia.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Tupla (é_válido, mensagem_erro)
        """
        file_path = Path(file_path)
        
        # Verificar se existe
        if not file_path.exists():
            return False, "Arquivo não encontrado"
        
        # Verificar se é arquivo
        if not file_path.is_file():
            return False, "Caminho não é um arquivo"
        
        # Verificar tamanho
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            size_mb = file_size / (1024 * 1024)
            max_mb = self.max_file_size / (1024 * 1024)
            return False, f"Arquivo muito grande ({size_mb:.1f}MB). Máximo: {max_mb:.1f}MB"
        
        if file_size == 0:
            return False, "Arquivo vazio"
        
        # Verificar extensão
        if not self.is_supported_format(file_path):
            return False, f"Formato não suportado: {file_path.suffix}"
        
        # Verificar tipo MIME se possível
        if self.magic:
            try:
                mime_type = self.magic.from_file(str(file_path))
                if not self._is_valid_mime_type(mime_type):
                    return False, f"Tipo de arquivo inválido: {mime_type}"
            except Exception as e:
                console.print(f"[yellow]⚠️  Erro ao verificar tipo MIME: {e}[/yellow]")
        
        # Verificar integridade básica
        is_valid, error = self._check_file_integrity(file_path)
        if not is_valid:
            return False, error
        
        return True, None
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """
        Verifica se o formato é suportado.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            True se suportado
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats
    
    def _is_valid_mime_type(self, mime_type: str) -> bool:
        """Verifica se o tipo MIME é válido."""
        valid_mime_types = [
            "audio/",
            "video/",
            "application/ogg",
            "application/x-matroska"
        ]
        
        return any(mime_type.startswith(valid) for valid in valid_mime_types)
    
    def _check_file_integrity(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Verifica integridade básica do arquivo.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Tupla (é_válido, mensagem_erro)
        """
        try:
            # Tentar ler os primeiros bytes
            with open(file_path, "rb") as f:
                header = f.read(12)
                
                if len(header) < 12:
                    return False, "Arquivo muito pequeno ou corrompido"
                
                # Verificar assinaturas conhecidas
                if file_path.suffix.lower() == ".mp3":
                    if not (header.startswith(b"ID3") or header[0:2] == b"\xff\xfb"):
                        return False, "Arquivo MP3 inválido"
                
                elif file_path.suffix.lower() == ".wav":
                    if not header.startswith(b"RIFF"):
                        return False, "Arquivo WAV inválido"
                
                elif file_path.suffix.lower() in [".mp4", ".m4a"]:
                    if not (b"ftyp" in header or b"moov" in header):
                        return False, "Arquivo MP4/M4A inválido"
            
            return True, None
            
        except Exception as e:
            return False, f"Erro ao ler arquivo: {str(e)}"
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict:
        """
        Obtém informações detalhadas do arquivo.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Dict com informações do arquivo
        """
        file_path = Path(file_path)
        
        info = {
            "path": str(file_path),
            "name": file_path.name,
            "extension": file_path.suffix.lower(),
            "size": file_path.stat().st_size if file_path.exists() else 0,
            "size_mb": (file_path.stat().st_size / (1024 * 1024)) if file_path.exists() else 0,
            "exists": file_path.exists(),
            "is_valid": False,
            "mime_type": None,
            "errors": []
        }
        
        # Validar arquivo
        is_valid, error = self.validate_file(file_path)
        info["is_valid"] = is_valid
        if error:
            info["errors"].append(error)
        
        # Obter tipo MIME
        if self.magic and file_path.exists():
            try:
                info["mime_type"] = self.magic.from_file(str(file_path))
            except:
                info["mime_type"] = mimetypes.guess_type(str(file_path))[0]
        
        return info


class URLValidator:
    """Validador de URLs para download."""
    
    def __init__(self):
        """Inicializa o validador de URLs."""
        self.supported_platforms = [
            "youtube.com",
            "youtu.be",
            "vimeo.com",
            "dailymotion.com",
            "soundcloud.com",
            "twitch.tv"
        ]
    
    def validate_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Valida uma URL.
        
        Args:
            url: URL para validar
            
        Returns:
            Tupla (é_válido, mensagem_erro)
        """
        # Verificar formato básico
        if not url or not isinstance(url, str):
            return False, "URL inválida"
        
        # Parse URL
        try:
            parsed = urlparse(url)
            
            if not parsed.scheme:
                return False, "URL deve começar com http:// ou https://"
            
            if parsed.scheme not in ["http", "https"]:
                return False, "Apenas URLs HTTP/HTTPS são suportadas"
            
            if not parsed.netloc:
                return False, "URL inválida - domínio não encontrado"
            
        except Exception as e:
            return False, f"URL mal formada: {str(e)}"
        
        # Verificar se é de plataforma suportada
        domain = parsed.netloc.lower().replace("www.", "")
        
        is_supported = any(platform in domain for platform in self.supported_platforms)
        
        if not is_supported:
            return False, f"Plataforma não suportada. Plataformas suportadas: {', '.join(self.supported_platforms)}"
        
        return True, None
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extrai ID do vídeo da URL.
        
        Args:
            url: URL do vídeo
            
        Returns:
            ID do vídeo ou None
        """
        # YouTube
        youtube_patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in youtube_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Vimeo
        vimeo_pattern = r'vimeo\.com\/(\d+)'
        match = re.search(vimeo_pattern, url)
        if match:
            return match.group(1)
        
        return None
    
    def get_platform(self, url: str) -> Optional[str]:
        """
        Identifica a plataforma da URL.
        
        Args:
            url: URL para identificar
            
        Returns:
            Nome da plataforma ou None
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        
        for platform in self.supported_platforms:
            if platform in domain:
                return platform.split(".")[0]  # Retorna nome sem .com
        
        return None


class SystemValidator:
    """Validador de requisitos do sistema."""
    
    @staticmethod
    def check_dependencies() -> Dict[str, Dict]:
        """
        Verifica todas as dependências do sistema.
        
        Returns:
            Dict com status de cada dependência
        """
        dependencies = {}
        
        # Python packages
        packages = [
            "whisper",
            "torch",
            "ffmpeg",
            "pydub",
            "rich",
            "yt_dlp"
        ]
        
        for package in packages:
            try:
                __import__(package)
                dependencies[package] = {
                    "installed": True,
                    "status": "OK",
                    "message": "Instalado"
                }
            except ImportError:
                dependencies[package] = {
                    "installed": False,
                    "status": "Erro",
                    "message": "Não instalado"
                }
        
        # FFmpeg
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                version_line = result.stdout.split("\n")[0]
                dependencies["ffmpeg_cli"] = {
                    "installed": True,
                    "status": "OK",
                    "message": version_line
                }
            else:
                raise Exception("FFmpeg retornou erro")
        except:
            dependencies["ffmpeg_cli"] = {
                "installed": False,
                "status": "Erro",
                "message": "FFmpeg não encontrado no PATH"
            }
        
        # GPU/CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                dependencies["cuda"] = {
                    "installed": True,
                    "status": "OK",
                    "message": f"CUDA disponível - {torch.cuda.get_device_name()}"
                }
            else:
                dependencies["cuda"] = {
                    "installed": False,
                    "status": "Info",
                    "message": "CUDA não disponível - usando CPU"
                }
        except:
            dependencies["cuda"] = {
                "installed": False,
                "status": "Info",
                "message": "Torch não instalado"
            }
        
        # Ollama
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                dependencies["ollama"] = {
                    "installed": True,
                    "status": "OK",
                    "message": "Ollama instalado"
                }
            else:
                raise Exception("Ollama não encontrado")
        except:
            dependencies["ollama"] = {
                "installed": False,
                "status": "Aviso",
                "message": "Ollama não instalado - resumos não disponíveis"
            }
        
        return dependencies
    
    @staticmethod
    def check_disk_space(required_gb: float = 5.0) -> Tuple[bool, Dict]:
        """
        Verifica espaço em disco disponível.
        
        Args:
            required_gb: Espaço mínimo necessário em GB
            
        Returns:
            Tupla (tem_espaço_suficiente, info_disco)
        """
        import shutil
        
        # Obter espaço do disco onde está o projeto
        stat = shutil.disk_usage(".")
        
        info = {
            "total_gb": stat.total / (1024**3),
            "used_gb": stat.used / (1024**3),
            "free_gb": stat.free / (1024**3),
            "used_percent": (stat.used / stat.total) * 100
        }
        
        has_space = info["free_gb"] >= required_gb
        
        return has_space, info
    
    @staticmethod
    def display_system_check():
        """Exibe verificação completa do sistema."""
        console.print("\n🔍 [bold]Verificação do Sistema[/bold]\n")
        
        # Verificar dependências
        deps = SystemValidator.check_dependencies()
        
        deps_table = Table(title="Dependências", show_lines=True)
        deps_table.add_column("Componente", style="cyan")
        deps_table.add_column("Status", style="white")
        deps_table.add_column("Mensagem", style="white")
        
        for name, info in deps.items():
            status_style = {
                "OK": "[green]✅ OK[/green]",
                "Erro": "[red]❌ Erro[/red]",
                "Aviso": "[yellow]⚠️  Aviso[/yellow]",
                "Info": "[blue]ℹ️  Info[/blue]"
            }.get(info["status"], info["status"])
            
            deps_table.add_row(
                name.replace("_", " ").title(),
                status_style,
                info["message"]
            )
        
        console.print(deps_table)
        
        # Verificar espaço em disco
        has_space, disk_info = SystemValidator.check_disk_space()
        
        console.print(f"\n💾 [bold]Espaço em Disco:[/bold]")
        console.print(f"   Total: {disk_info['total_gb']:.1f} GB")
        console.print(f"   Usado: {disk_info['used_gb']:.1f} GB ({disk_info['used_percent']:.1f}%)")
        console.print(f"   Livre: {disk_info['free_gb']:.1f} GB")
        
        if not has_space:
            console.print(f"   [yellow]⚠️  Espaço em disco baixo![/yellow]")
        else:
            console.print(f"   [green]✅ Espaço suficiente[/green]")
        
        # Verificar configurações
        console.print(f"\n⚙️  [bold]Configurações Atuais:[/bold]")
        console.print(f"   Modelo Whisper: {WHISPER_CONFIG['model_size']}")
        console.print(f"   Dispositivo: {WHISPER_CONFIG['device']}")
        console.print(f"   Idioma padrão: {WHISPER_CONFIG['language']}")
        
        # Resumo
        all_ok = all(d["status"] in ["OK", "Info"] for d in deps.values()) and has_space
        
        if all_ok:
            console.print(f"\n[green]✅ Sistema pronto para uso![/green]")
        else:
            console.print(f"\n[yellow]⚠️  Alguns componentes precisam de atenção[/yellow]")


def validate_transcription_params(params: Dict) -> Tuple[bool, List[str]]:
    """
    Valida parâmetros de transcrição.
    
    Args:
        params: Parâmetros para validar
        
    Returns:
        Tupla (é_válido, lista_de_erros)
    """
    errors = []
    
    # Validar modelo
    valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    if "model_size" in params and params["model_size"] not in valid_models:
        errors.append(f"Modelo inválido: {params['model_size']}")
    
    # Validar idioma
    if "language" in params:
        # Lista básica de códigos de idioma válidos
        valid_languages = [
            "pt", "en", "es", "fr", "de", "it", "ja", "ko", "zh", "ru",
            "ar", "hi", "nl", "pl", "tr", "sv", "da", "no", "fi"
        ]
        if params["language"] not in valid_languages:
            errors.append(f"Idioma não suportado: {params['language']}")
    
    # Validar temperatura
    if "temperature" in params:
        temp = params["temperature"]
        if isinstance(temp, (int, float)):
            if temp < 0 or temp > 1:
                errors.append(f"Temperatura deve estar entre 0 e 1: {temp}")
        elif isinstance(temp, list):
            for t in temp:
                if t < 0 or t > 1:
                    errors.append(f"Temperatura deve estar entre 0 e 1: {t}")
    
    # Validar dispositivo
    if "device" in params and params["device"] not in ["cpu", "cuda"]:
        errors.append(f"Dispositivo inválido: {params['device']}")
    
    return len(errors) == 0, errors