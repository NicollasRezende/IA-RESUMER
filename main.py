#!/usr/bin/env python3
"""
Sistema de Transcrição de Áudio - MVP
Ponto de entrada principal da aplicação.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.transcriber import WhisperTranscriber
from src.core.audio_processor import AudioProcessor
from src.core.config import (
    SUPPORTED_AUDIO_FORMATS, 
    WHISPER_CONFIG,
    get_model_info
)

console = Console()


def print_banner():
    """Exibe o banner do sistema."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║         🎙️  SISTEMA DE TRANSCRIÇÃO DE ÁUDIO 🎙️           ║
    ║                   Powered by Whisper                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold cyan"))


def print_supported_formats():
    """Exibe os formatos suportados."""
    console.print("\n📁 [bold]Formatos suportados:[/bold]")
    console.print("   " + ", ".join(SUPPORTED_AUDIO_FORMATS))


def transcribe_file(
    file_path: str,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    output_format: str = "json",
    device: Optional[str] = None,
    detect_language: bool = False
) -> Path:
    """
    Transcreve um arquivo de áudio.
    
    Args:
        file_path: Caminho do arquivo
        model_size: Tamanho do modelo Whisper
        language: Idioma do áudio
        output_format: Formato de saída
        device: Dispositivo (cpu/cuda)
        detect_language: Se deve detectar o idioma
        
    Returns:
        Path do arquivo de transcrição
    """
    file_path = Path(file_path)
    
    # Verificar se o arquivo existe
    if not file_path.exists():
        console.print(f"[bold red]❌ Arquivo não encontrado: {file_path}[/bold red]")
        sys.exit(1)
    
    # Inicializar processador de áudio
    audio_processor = AudioProcessor()
    
    # Verificar formato
    if not audio_processor.is_supported_format(file_path):
        console.print(f"[bold red]❌ Formato não suportado: {file_path.suffix}[/bold red]")
        print_supported_formats()
        sys.exit(1)
    
    # Obter informações do áudio
    console.print(f"\n📊 [bold]Informações do arquivo:[/bold]")
    audio_info = audio_processor.get_audio_info(file_path)
    
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Campo", style="cyan")
    info_table.add_column("Valor", style="white")
    
    for key, value in audio_info.items():
        if key not in ["filename"]:  # Já mostramos o nome
            info_table.add_row(key.replace("_", " ").title() + ":", str(value))
    
    console.print(info_table)
    
    # Processar arquivo se necessário
    processed_path = file_path
    
    # Se for vídeo, extrair áudio
    if file_path.suffix.lower() in ['.mp4', '.mkv', '.avi', '.webm']:
        console.print(f"\n🎬 Arquivo de vídeo detectado, extraindo áudio...")
        processed_path = audio_processor.extract_audio_from_video(file_path)
    
    # Converter para WAV se necessário
    elif file_path.suffix.lower() != '.wav':
        processed_path = audio_processor.convert_to_wav(file_path)
    
    # Inicializar transcritor
    console.print(f"\n🚀 Inicializando sistema de transcrição...")
    transcriber = WhisperTranscriber(model_size=model_size, device=device)
    
    # Detectar idioma se solicitado
    if detect_language:
        detected_lang = transcriber.detect_language(processed_path)
        language = language or detected_lang
    
    # Transcrever
    console.print(f"\n🎤 Iniciando transcrição...")
    result = transcriber.transcribe_with_segments(
        processed_path,
        language=language
    )
    
    # Salvar resultado
    output_path = transcriber.save_transcription(result, format=output_format)
    
    # Mostrar estatísticas
    console.print(f"\n📈 [bold]Estatísticas:[/bold]")
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Métrica", style="cyan")
    stats_table.add_column("Valor", style="white")
    
    metadata = result["metadata"]
    stats_table.add_row("Duração do áudio:", f"{metadata['duration']:.1f}s")
    stats_table.add_row("Tempo de processamento:", f"{metadata['processing_time']:.1f}s")
    stats_table.add_row("Velocidade:", f"{metadata['duration']/metadata['processing_time']:.1f}x tempo real")
    stats_table.add_row("Palavras transcritas:", str(len(result["text"].split())))
    stats_table.add_row("Segmentos:", str(len(result["segments"])))
    
    console.print(stats_table)
    
    # Limpar arquivos temporários
    if processed_path != file_path:
        audio_processor.clean_temp_files()
    
    # Mostrar preview da transcrição
    console.print(f"\n📝 [bold]Preview da transcrição:[/bold]")
    preview_text = result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"]
    console.print(Panel(preview_text, style="green"))
    
    return output_path


def list_models():
    """Lista os modelos disponíveis."""
    console.print("\n🤖 [bold]Modelos Whisper disponíveis:[/bold]\n")
    
    table = Table(title="Modelos e Requisitos", show_lines=True)
    table.add_column("Modelo", style="cyan", justify="center")
    table.add_column("Tamanho", style="green", justify="center")
    table.add_column("VRAM", style="yellow", justify="center")
    table.add_column("Velocidade Relativa", style="magenta", justify="center")
    table.add_column("Qualidade", style="blue", justify="center")
    
    models = {
        "tiny": {"quality": "★★☆☆☆", "use_case": "Testes rápidos"},
        "base": {"quality": "★★★☆☆", "use_case": "Transcrições básicas"},
        "small": {"quality": "★★★★☆", "use_case": "Melhor custo-benefício"},
        "medium": {"quality": "★★★★★", "use_case": "Alta qualidade"},
        "large": {"quality": "★★★★★", "use_case": "Máxima qualidade"},
    }
    
    for model_name, extra_info in models.items():
        info = get_model_info(model_name)
        table.add_row(
            model_name,
            info["size"],
            info["vram"],
            f"{info['relative_speed']}x",
            extra_info["quality"]
        )
    
    console.print(table)
    console.print(f"\n💡 [bold]Modelo atual:[/bold] {WHISPER_CONFIG['model_size']}")
    console.print(f"💡 [bold]Dispositivo:[/bold] {WHISPER_CONFIG['device']}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Sistema de Transcrição de Áudio usando OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "file",
        nargs="?",
        help="Arquivo de áudio/vídeo para transcrever"
    )
    
    parser.add_argument(
        "-m", "--model",
        default=None,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help=f"Modelo Whisper a usar (padrão: {WHISPER_CONFIG['model_size']})"
    )
    
    parser.add_argument(
        "-l", "--language",
        default=None,
        help=f"Idioma do áudio (padrão: {WHISPER_CONFIG['language']})"
    )
    
    parser.add_argument(
        "-f", "--format",
        default="json",
        choices=["json", "txt", "srt", "vtt"],
        help="Formato de saída (padrão: json)"
    )
    
    parser.add_argument(
        "-d", "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Dispositivo para execução"
    )
    
    parser.add_argument(
        "--detect-language",
        action="store_true",
        help="Detectar idioma automaticamente"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Listar modelos disponíveis"
    )
    
    parser.add_argument(
        "--formats",
        action="store_true",
        help="Mostrar formatos suportados"
    )
    
    args = parser.parse_args()
    
    # Mostrar banner
    print_banner()
    
    # Comandos informativos
    if args.list_models:
        list_models()
        return
    
    if args.formats:
        print_supported_formats()
        return
    
    # Verificar se foi fornecido um arquivo
    if not args.file:
        console.print("[bold red]❌ Por favor, forneça um arquivo para transcrever![/bold red]")
        console.print("\n[bold]Uso:[/bold] python main.py <arquivo> [opções]")
        console.print("\nPara mais informações: python main.py --help")
        sys.exit(1)
    
    # Transcrever arquivo
    try:
        output_path = transcribe_file(
            args.file,
            model_size=args.model,
            language=args.language,
            output_format=args.format,
            device=args.device,
            detect_language=args.detect_language
        )
        
        console.print(f"\n✅ [bold green]Transcrição concluída com sucesso![/bold green]")
        console.print(f"📄 Arquivo salvo em: [bold]{output_path}[/bold]")
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Processo interrompido pelo usuário[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Erro durante a transcrição: {e}[/bold red]")
        console.print("\n[dim]Para reportar bugs, visite: https://github.com/seu-usuario/audio-transcription-system[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()