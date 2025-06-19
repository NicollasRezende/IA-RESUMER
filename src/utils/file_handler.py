"""
Utilitários para manipulação de arquivos e cache.
"""
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from rich.console import Console
from rich.table import Table
from rich.progress import track

from ..core.config import (
    UPLOAD_DIR,
    TRANSCRIPT_DIR,
    TEMP_DIR,
    PERFORMANCE_CONFIG
)

console = Console()


class FileHandler:
    """Gerenciador de arquivos do sistema."""
    
    def __init__(self):
        """Inicializa o gerenciador de arquivos."""
        self.upload_dir = UPLOAD_DIR
        self.transcript_dir = TRANSCRIPT_DIR
        self.temp_dir = TEMP_DIR
        self.cache_enabled = PERFORMANCE_CONFIG["cache_enabled"]
        self.cache_ttl = PERFORMANCE_CONFIG["cache_ttl"]
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Garante que os diretórios necessários existam."""
        for directory in [self.upload_dir, self.transcript_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_upload(self, file_path: Union[str, Path], filename: Optional[str] = None) -> Path:
        """
        Salva um arquivo no diretório de uploads.
        
        Args:
            file_path: Caminho do arquivo origem
            filename: Nome desejado (opcional)
            
        Returns:
            Path do arquivo salvo
        """
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {source}")
        
        if filename is None:
            # Gerar nome único baseado em timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{source.stem}_{timestamp}{source.suffix}"
        
        destination = self.upload_dir / filename
        
        # Copiar arquivo
        shutil.copy2(source, destination)
        console.print(f"📁 Arquivo salvo em uploads: [bold]{destination.name}[/bold]")
        
        return destination
    
    def get_file_hash(self, file_path: Union[str, Path]) -> str:
        """
        Calcula o hash SHA256 de um arquivo.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Hash hexadecimal do arquivo
        """
        file_path = Path(file_path)
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def check_cache(self, file_hash: str, operation: str = "transcription") -> Optional[Dict]:
        """
        Verifica se existe resultado em cache para o arquivo.
        
        Args:
            file_hash: Hash do arquivo
            operation: Tipo de operação (transcription, summary, etc)
            
        Returns:
            Resultado em cache ou None
        """
        if not self.cache_enabled:
            return None
        
        cache_file = self.transcript_dir / f".cache_{operation}_{file_hash}.json"
        
        if not cache_file.exists():
            return None
        
        # Verificar TTL
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > timedelta(seconds=self.cache_ttl):
            console.print(f"🕒 Cache expirado para {operation}")
            cache_file.unlink()
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            console.print(f"✅ Usando cache para {operation}")
            return data
        except Exception as e:
            console.print(f"[yellow]⚠️  Erro ao ler cache: {e}[/yellow]")
            return None
    
    def save_cache(self, file_hash: str, data: Dict, operation: str = "transcription"):
        """
        Salva resultado em cache.
        
        Args:
            file_hash: Hash do arquivo
            data: Dados para cachear
            operation: Tipo de operação
        """
        if not self.cache_enabled:
            return
        
        cache_file = self.transcript_dir / f".cache_{operation}_{file_hash}.json"
        
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console.print(f"[yellow]⚠️  Erro ao salvar cache: {e}[/yellow]")
    
    def list_transcriptions(self, limit: int = 10) -> List[Dict]:
        """
        Lista as transcrições mais recentes.
        
        Args:
            limit: Número máximo de resultados
            
        Returns:
            Lista de informações das transcrições
        """
        transcriptions = []
        
        # Buscar arquivos JSON (excluindo cache)
        json_files = [f for f in self.transcript_dir.glob("*.json") 
                     if not f.name.startswith(".cache_")]
        
        # Ordenar por data de modificação (mais recente primeiro)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for json_file in json_files[:limit]:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                metadata = data.get("metadata", {})
                
                transcriptions.append({
                    "file": json_file.name,
                    "path": json_file,
                    "original_file": metadata.get("file", "N/A"),
                    "duration": metadata.get("duration", 0),
                    "language": metadata.get("language", "N/A"),
                    "model": metadata.get("model", "N/A"),
                    "created": datetime.fromtimestamp(json_file.stat().st_mtime),
                    "size": json_file.stat().st_size / 1024  # KB
                })
            except Exception as e:
                console.print(f"[yellow]⚠️  Erro ao ler {json_file.name}: {e}[/yellow]")
        
        return transcriptions
    
    def display_transcription_list(self, transcriptions: List[Dict]):
        """
        Exibe lista de transcrições em tabela formatada.
        
        Args:
            transcriptions: Lista de transcrições
        """
        if not transcriptions:
            console.print("[yellow]Nenhuma transcrição encontrada.[/yellow]")
            return
        
        table = Table(title="Transcrições Recentes", show_lines=True)
        table.add_column("Arquivo", style="cyan")
        table.add_column("Original", style="white")
        table.add_column("Duração", style="green")
        table.add_column("Idioma", style="yellow")
        table.add_column("Modelo", style="blue")
        table.add_column("Data", style="magenta")
        table.add_column("Tamanho", style="white")
        
        for trans in transcriptions:
            table.add_row(
                trans["file"][:30] + "..." if len(trans["file"]) > 30 else trans["file"],
                trans["original_file"],
                f"{trans['duration']:.1f}s",
                trans["language"],
                trans["model"],
                trans["created"].strftime("%Y-%m-%d %H:%M"),
                f"{trans['size']:.1f} KB"
            )
        
        console.print(table)
    
    def clean_old_files(self, days: int = 7, dry_run: bool = True) -> Dict[str, int]:
        """
        Remove arquivos antigos.
        
        Args:
            days: Arquivos mais antigos que isso serão removidos
            dry_run: Se True, apenas simula a remoção
            
        Returns:
            Dict com contagem de arquivos removidos por tipo
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        removed_count = {
            "uploads": 0,
            "transcripts": 0,
            "temp": 0,
            "cache": 0
        }
        
        console.print(f"\n🧹 {'Simulando limpeza' if dry_run else 'Limpando'} arquivos mais antigos que {days} dias...")
        
        # Limpar uploads antigos
        for file in self.upload_dir.glob("*"):
            if file.is_file() and datetime.fromtimestamp(file.stat().st_mtime) < cutoff_date:
                if not dry_run:
                    file.unlink()
                removed_count["uploads"] += 1
        
        # Limpar transcrições antigas
        for file in self.transcript_dir.glob("*.json"):
            if file.is_file() and datetime.fromtimestamp(file.stat().st_mtime) < cutoff_date:
                if not dry_run:
                    file.unlink()
                removed_count["transcripts"] += 1
        
        # Limpar cache
        for file in self.transcript_dir.glob(".cache_*"):
            if file.is_file() and datetime.fromtimestamp(file.stat().st_mtime) < cutoff_date:
                if not dry_run:
                    file.unlink()
                removed_count["cache"] += 1
        
        # Sempre limpar arquivos temporários
        for file in self.temp_dir.glob("*"):
            if file.is_file():
                if not dry_run:
                    file.unlink()
                removed_count["temp"] += 1
        
        # Mostrar resultados
        total = sum(removed_count.values())
        
        if dry_run:
            console.print(f"\n[yellow]Modo simulação - nenhum arquivo foi removido[/yellow]")
        
        console.print(f"\nResumo da limpeza:")
        console.print(f"  📁 Uploads: {removed_count['uploads']} arquivos")
        console.print(f"  📄 Transcrições: {removed_count['transcripts']} arquivos")
        console.print(f"  💾 Cache: {removed_count['cache']} arquivos")
        console.print(f"  🗑️  Temporários: {removed_count['temp']} arquivos")
        console.print(f"  [bold]Total: {total} arquivos[/bold]")
        
        return removed_count
    
    def get_storage_stats(self) -> Dict[str, Dict]:
        """
        Obtém estatísticas de uso de armazenamento.
        
        Returns:
            Dict com estatísticas por diretório
        """
        stats = {}
        
        for name, directory in [
            ("uploads", self.upload_dir),
            ("transcripts", self.transcript_dir),
            ("temp", self.temp_dir)
        ]:
            total_size = 0
            file_count = 0
            
            for file in directory.glob("**/*"):
                if file.is_file():
                    total_size += file.stat().st_size
                    file_count += 1
            
            stats[name] = {
                "total_size_mb": total_size / (1024 * 1024),
                "file_count": file_count,
                "path": str(directory)
            }
        
        return stats
    
    def display_storage_stats(self):
        """Exibe estatísticas de armazenamento."""
        stats = self.get_storage_stats()
        
        console.print("\n📊 [bold]Estatísticas de Armazenamento[/bold]")
        
        table = Table(show_lines=True)
        table.add_column("Diretório", style="cyan")
        table.add_column("Arquivos", style="green")
        table.add_column("Tamanho", style="yellow")
        table.add_column("Caminho", style="white")
        
        total_files = 0
        total_size = 0
        
        for name, data in stats.items():
            table.add_row(
                name.capitalize(),
                str(data["file_count"]),
                f"{data['total_size_mb']:.2f} MB",
                data["path"]
            )
            total_files += data["file_count"]
            total_size += data["total_size_mb"]
        
        console.print(table)
        
        console.print(f"\n[bold]Total:[/bold] {total_files} arquivos, {total_size:.2f} MB")
    
    def export_transcription(
        self,
        transcription_path: Union[str, Path],
        export_format: str = "txt",
        include_metadata: bool = False
    ) -> Path:
        """
        Exporta transcrição para diferentes formatos.
        
        Args:
            transcription_path: Caminho do arquivo JSON de transcrição
            export_format: Formato de exportação (txt, md, docx, pdf)
            include_metadata: Se deve incluir metadados
            
        Returns:
            Path do arquivo exportado
        """
        transcription_path = Path(transcription_path)
        
        if not transcription_path.exists():
            raise FileNotFoundError(f"Transcrição não encontrada: {transcription_path}")
        
        # Carregar transcrição
        with open(transcription_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Definir caminho de saída
        export_path = transcription_path.parent / f"{transcription_path.stem}_export.{export_format}"
        
        if export_format == "txt":
            self._export_to_txt(data, export_path, include_metadata)
        elif export_format == "md":
            self._export_to_markdown(data, export_path, include_metadata)
        else:
            raise ValueError(f"Formato não suportado: {export_format}")
        
        console.print(f"✅ Exportado para: [bold]{export_path}[/bold]")
        return export_path
    
    def _export_to_txt(self, data: Dict, output_path: Path, include_metadata: bool):
        """Exporta para formato TXT."""
        with open(output_path, "w", encoding="utf-8") as f:
            if include_metadata:
                metadata = data.get("metadata", {})
                f.write("INFORMAÇÕES DA TRANSCRIÇÃO\n")
                f.write("=" * 50 + "\n")
                f.write(f"Arquivo: {metadata.get('file', 'N/A')}\n")
                f.write(f"Duração: {metadata.get('duration', 0):.1f} segundos\n")
                f.write(f"Idioma: {metadata.get('language', 'N/A')}\n")
                f.write(f"Modelo: {metadata.get('model', 'N/A')}\n")
                f.write("=" * 50 + "\n\n")
            
            f.write("TRANSCRIÇÃO\n")
            f.write("=" * 50 + "\n")
            f.write(data.get("text", ""))
    
    def _export_to_markdown(self, data: Dict, output_path: Path, include_metadata: bool):
        """Exporta para formato Markdown."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Transcrição de Áudio\n\n")
            
            if include_metadata:
                metadata = data.get("metadata", {})
                f.write("## Informações\n\n")
                f.write(f"- **Arquivo**: {metadata.get('file', 'N/A')}\n")
                f.write(f"- **Duração**: {metadata.get('duration', 0):.1f} segundos\n")
                f.write(f"- **Idioma**: {metadata.get('language', 'N/A')}\n")
                f.write(f"- **Modelo**: {metadata.get('model', 'N/A')}\n")
                f.write(f"- **Processamento**: {metadata.get('processing_time', 0):.1f} segundos\n")
                f.write("\n")
            
            f.write("## Transcrição\n\n")
            f.write(data.get("text", ""))
            
            # Se houver segmentos com timestamps
            segments = data.get("segments", [])
            if segments and len(segments) > 1:
                f.write("\n\n## Segmentos com Timestamps\n\n")
                for segment in segments:
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    text = segment.get("text", "").strip()
                    f.write(f"**[{start:.1f}s - {end:.1f}s]** {text}\n\n")