#!/usr/bin/env python3
"""
Sistema de Transcrição e Resumo de Áudio - Versão Aprimorada
Ponto de entrada principal com todas as funcionalidades.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, List
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.transcriber import WhisperTranscriber
from src.core.summarizer import OllamaSummarizer, SummaryType
from src.core.audio_processor import AudioProcessor
from src.core.config import (
    SUPPORTED_AUDIO_FORMATS, 
    WHISPER_CONFIG,
    get_model_info
)
from src.utils.file_handler import FileHandler
from src.utils.validators import MediaValidator, URLValidator, SystemValidator

console = Console()


def print_banner():
    """Exibe o banner do sistema."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║        🎙️  SISTEMA AVANÇADO DE TRANSCRIÇÃO E RESUMO 🎙️              ║
    ║                    Powered by Whisper + Ollama                        ║
    ║                          Versão 2.0                                   ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold cyan"))


def interactive_mode():
    """Modo interativo com menu completo."""
    while True:
        console.print("\n[bold cyan]MENU PRINCIPAL[/bold cyan]")
        console.print("1. 🎤 Transcrever arquivo")
        console.print("2. 📝 Gerar resumo de transcrição")
        console.print("3. 📊 Ver transcrições recentes")
        console.print("4. 🔍 Verificar sistema")
        console.print("5. 🧹 Limpar arquivos antigos")
        console.print("6. 💾 Estatísticas de armazenamento")
        console.print("7. ❌ Sair")
        
        choice = Prompt.ask("\nEscolha uma opção", choices=["1", "2", "3", "4", "5", "6", "7"])
        
        if choice == "1":
            transcribe_interactive()
        elif choice == "2":
            summarize_interactive()
        elif choice == "3":
            list_transcriptions_interactive()
        elif choice == "4":
            SystemValidator.display_system_check()
        elif choice == "5":
            clean_files_interactive()
        elif choice == "6":
            file_handler = FileHandler()
            file_handler.display_storage_stats()
        elif choice == "7":
            console.print("\n[green]Até logo! 👋[/green]")
            break


def transcribe_interactive():
    """Modo interativo de transcrição."""
    console.print("\n[bold]🎤 TRANSCRIÇÃO DE ÁUDIO[/bold]")
    
    # Obter arquivo
    file_path = Prompt.ask("Caminho do arquivo")
    file_path = Path(file_path.strip('"').strip("'"))
    
    # Validar arquivo
    validator = MediaValidator()
    is_valid, error = validator.validate_file(file_path)
    
    if not is_valid:
        console.print(f"[red]❌ Erro: {error}[/red]")
        return
    
    # Opções de transcrição
    console.print("\n[bold]Opções de transcrição:[/bold]")
    
    # Modelo
    model_size = Prompt.ask(
        "Modelo Whisper",
        choices=["tiny", "base", "small", "medium", "large"],
        default=WHISPER_CONFIG["model_size"]
    )
    
    # Idioma
    detect_language = Confirm.ask("Detectar idioma automaticamente?", default=True)
    language = None
    if not detect_language:
        language = Prompt.ask("Idioma (pt, en, es, etc)", default="pt")
    
    # Qualidade
    use_enhanced = Confirm.ask("Usar transcrição aprimorada? (mais lenta, melhor qualidade)", default=True)
    
    # Transcrever
    try:
        if use_enhanced:
            transcriber = WhisperTranscriber(model_size=model_size)
            result = transcriber.transcribe_enhanced(
                file_path,
                language=language,
                detect_language=detect_language
            )
            
            # Mostrar relatório de qualidade
            transcriber.generate_quality_report(result)
        else:
            # Usar transcritor básico
            from src.core.transcriber import WhisperTranscriber
            transcriber = WhisperTranscriber(model_size=model_size)
            result = transcriber.transcribe_with_segments(
                file_path,
                language=language
            )
        
        # Salvar resultado
        output_path = transcriber.save_transcription(result)
        
        # Perguntar se quer gerar resumo
        if Confirm.ask("\nGerar resumo automático?", default=True):
            generate_summary_for_transcription(output_path)
            
    except Exception as e:
        console.print(f"[red]❌ Erro durante transcrição: {e}[/red]")


def summarize_interactive():
    """Modo interativo para gerar resumo."""
    console.print("\n[bold]📝 GERAR RESUMO[/bold]")
    
    # Listar transcrições recentes
    file_handler = FileHandler()
    transcriptions = file_handler.list_transcriptions(limit=10)
    
    if not transcriptions:
        console.print("[yellow]Nenhuma transcrição encontrada.[/yellow]")
        return
    
    # Mostrar lista
    console.print("\n[bold]Transcrições disponíveis:[/bold]")
    for i, trans in enumerate(transcriptions, 1):
        console.print(f"{i}. {trans['original_file']} - {trans['created'].strftime('%Y-%m-%d %H:%M')}")
    
    # Escolher transcrição
    choice = Prompt.ask("\nEscolha uma transcrição", choices=[str(i) for i in range(1, len(transcriptions)+1)])
    selected = transcriptions[int(choice)-1]
    
    generate_summary_for_transcription(selected['path'])


def generate_summary_for_transcription(transcription_path: Path):
    """Gera resumo para uma transcrição."""
    # Carregar transcrição
    with open(transcription_path, "r", encoding="utf-8") as f:
        transcription_data = json.load(f)
    
    # Escolher tipos de resumo
    console.print("\n[bold]Tipos de resumo disponíveis:[/bold]")
    summary_types_map = {
        "1": (SummaryType.EXECUTIVE, "Resumo Executivo"),
        "2": (SummaryType.BULLET_POINTS, "Pontos Principais"),
        "3": (SummaryType.TECHNICAL, "Resumo Técnico"),
        "4": (SummaryType.SIMPLIFIED, "Versão Simplificada"),
        "5": (SummaryType.COMPREHENSIVE, "Análise Completa"),
        "6": (SummaryType.ACADEMIC, "Resumo Acadêmico")
    }
    
    for key, (_, name) in summary_types_map.items():
        console.print(f"{key}. {name}")
    
    choices = Prompt.ask(
        "\nEscolha os tipos de resumo (separe por vírgula, ex: 1,2,3)",
        default="1,2"
    ).split(",")
    
    selected_types = [summary_types_map[c.strip()][0] for c in choices if c.strip() in summary_types_map]
    
    # Gerar resumos
    try:
        summarizer = OllamaSummarizer()
        results = summarizer.summarize_transcription(
            transcription_data,
            summary_types=selected_types
        )
        
        # Exibir resultados
        summarizer.display_summary_results(results)
        
        # Salvar resumos
        summary_path = transcription_path.parent / f"{transcription_path.stem}_summaries.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        console.print(f"\n💾 Resumos salvos em: [bold]{summary_path}[/bold]")
        
        # Perguntar se quer insights
        if Confirm.ask("\nGerar insights adicionais?", default=False):
            insights = summarizer.generate_insights(
                transcription_data["text"],
                domain=Prompt.ask("Domínio do conteúdo", default="general", 
                                choices=["general", "tech", "business", "academic"])
            )
            
            console.print("\n[bold]🔍 INSIGHTS:[/bold]")
            console.print(Panel(insights["insights"], border_style="cyan"))
            
    except Exception as e:
        console.print(f"[red]❌ Erro ao gerar resumo: {e}[/red]")


def list_transcriptions_interactive():
    """Lista transcrições de forma interativa."""
    file_handler = FileHandler()
    
    limit = Prompt.ask("Quantas transcrições mostrar?", default="10")
    transcriptions = file_handler.list_transcriptions(limit=int(limit))
    
    if transcriptions:
        file_handler.display_transcription_list(transcriptions)
        
        # Opções adicionais
        if Confirm.ask("\nExportar alguma transcrição?", default=False):
            choice = Prompt.ask("Número da transcrição", 
                              choices=[str(i) for i in range(1, len(transcriptions)+1)])
            
            selected = transcriptions[int(choice)-1]
            format = Prompt.ask("Formato de exportação", choices=["txt", "md"], default="md")
            
            export_path = file_handler.export_transcription(
                selected['path'],
                export_format=format,
                include_metadata=True
            )
            
            console.print(f"✅ Exportado para: [bold]{export_path}[/bold]")
    else:
        console.print("[yellow]Nenhuma transcrição encontrada.[/yellow]")


def clean_files_interactive():
    """Limpeza interativa de arquivos."""
    file_handler = FileHandler()
    
    days = Prompt.ask("Remover arquivos mais antigos que quantos dias?", default="7")
    
    # Primeiro fazer simulação
    console.print("\n[yellow]Simulando limpeza...[/yellow]")
    removed = file_handler.clean_old_files(days=int(days), dry_run=True)
    
    if sum(removed.values()) > 0:
        if Confirm.ask("\nConfirmar remoção?", default=False):
            removed = file_handler.clean_old_files(days=int(days), dry_run=False)
            console.print("\n[green]✅ Limpeza concluída![/green]")
    else:
        console.print("\n[green]Nenhum arquivo para remover.[/green]")


def transcribe_file_enhanced(
    file_path: str,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    output_format: str = "json",
    device: Optional[str] = None,
    detect_language: bool = False,
    quality_mode: str = "enhanced",
    generate_summary: bool = False,
    summary_types: Optional[List[str]] = None
) -> Path:
    """
    Transcreve um arquivo com todas as opções avançadas.
    
    Args:
        file_path: Caminho do arquivo
        model_size: Tamanho do modelo
        language: Idioma
        output_format: Formato de saída
        device: Dispositivo (cpu/cuda)
        detect_language: Detectar idioma automaticamente
        quality_mode: Modo de qualidade (basic, enhanced, auto)
        generate_summary: Se deve gerar resumo automático
        summary_types: Tipos de resumo para gerar
        
    Returns:
        Path do arquivo de transcrição
    """
    file_path = Path(file_path)
    
    # Validar arquivo
    validator = MediaValidator()
    is_valid, error = validator.validate_file(file_path)
    
    if not is_valid:
        console.print(f"[bold red]❌ {error}[/bold red]")
        sys.exit(1)
    
    # Verificar cache
    file_handler = FileHandler()
    file_hash = file_handler.get_file_hash(file_path)
    
    cached_result = file_handler.check_cache(file_hash, "transcription")
    if cached_result:
        console.print("\n[green]✅ Usando transcrição em cache[/green]")
        output_path = Path(cached_result.get("output_path", ""))
        
        if generate_summary and output_path.exists():
            generate_summary_for_transcription(output_path)
        
        return output_path
    
    # Escolher modo de transcrição
    if quality_mode == "auto":
        # Decidir baseado no tamanho do arquivo
        audio_processor = AudioProcessor()
        info = audio_processor.get_audio_info(file_path)
        duration = info.get("duration", 0)
        
        # Usar enhanced para arquivos menores que 10 minutos
        quality_mode = "enhanced" if duration < 600 else "basic"
        console.print(f"[cyan]Modo automático: usando transcrição {quality_mode}[/cyan]")
    
    # Transcrever
    if quality_mode == "enhanced":
        transcriber = WhisperTranscriber(model_size=model_size, device=device)
        
        # Tentar com múltiplos modelos se especificado
        if model_size == "auto":
            result = transcriber.transcribe_with_fallback(
                file_path,
                models=["small", "medium", "large"],
                language=language
            )
        else:
            result = transcriber.transcribe_enhanced(
                file_path,
                language=language,
                detect_language=detect_language
            )
        
        # Mostrar relatório de qualidade
        transcriber.generate_quality_report(result)
    else:
        # Modo básico
        from src.core.transcriber import WhisperTranscriber
        transcriber = WhisperTranscriber(model_size=model_size, device=device)
        
        if detect_language:
            language = transcriber.detect_language(file_path)
        
        result = transcriber.transcribe_with_segments(
            file_path,
            language=language
        )
    
    # Salvar transcrição
    output_path = transcriber.save_transcription(result, format=output_format)
    
    # Salvar em cache
    file_handler.save_cache(file_hash, {
        "output_path": str(output_path),
        "result": result
    }, "transcription")
    
    # Gerar resumo se solicitado
    if generate_summary:
        try:
            summarizer = OllamaSummarizer()
            
            # Converter tipos de string para enum
            if summary_types:
                summary_type_enums = []
                for st in summary_types:
                    try:
                        summary_type_enums.append(SummaryType[st.upper()])
                    except:
                        console.print(f"[yellow]Tipo de resumo inválido: {st}[/yellow]")
            else:
                summary_type_enums = [SummaryType.EXECUTIVE, SummaryType.BULLET_POINTS]
            
            summary_results = summarizer.summarize_transcription(
                result,
                summary_types=summary_type_enums
            )
            
            summarizer.display_summary_results(summary_results)
            
            # Salvar resumos
            summary_path = output_path.parent / f"{output_path.stem}_summaries.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_results, f, ensure_ascii=False, indent=2)
            
            console.print(f"\n💾 Resumos salvos em: [bold]{summary_path}[/bold]")
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Erro ao gerar resumo: {e}[/yellow]")
            console.print("[yellow]   Verifique se o Ollama está instalado e rodando[/yellow]")
    
    return output_path


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Sistema Avançado de Transcrição e Resumo de Áudio",
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
        choices=["tiny", "base", "small", "medium", "large", "auto"],
        help="Modelo Whisper (auto = escolhe automaticamente)"
    )
    
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Idioma do áudio"
    )
    
    parser.add_argument(
        "-f", "--format",
        default="json",
        choices=["json", "txt", "srt", "vtt"],
        help="Formato de saída"
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
        "-q", "--quality",
        default="enhanced",
        choices=["basic", "enhanced", "auto"],
        help="Modo de qualidade da transcrição"
    )
    
    parser.add_argument(
        "-s", "--summarize",
        action="store_true",
        help="Gerar resumo automático após transcrição"
    )
    
    parser.add_argument(
        "--summary-types",
        nargs="+",
        choices=["executive", "bullet_points", "technical", "simplified", "comprehensive", "academic"],
        help="Tipos de resumo para gerar"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Modo interativo"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verificar sistema"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="Listar transcrições recentes"
    )
    
    args = parser.parse_args()
    
    # Mostrar banner
    print_banner()
    
    # Comandos especiais
    if args.check:
        SystemValidator.display_system_check()
        return
    
    if args.list:
        file_handler = FileHandler()
        transcriptions = file_handler.list_transcriptions()
        file_handler.display_transcription_list(transcriptions)
        return
    
    if args.interactive:
        interactive_mode()
        return
    
    # Verificar se foi fornecido arquivo
    if not args.file:
        console.print("[bold]Nenhum arquivo especificado. Entrando em modo interativo...[/bold]\n")
        interactive_mode()
        return
    
    # Transcrever arquivo
    try:
        output_path = transcribe_file_enhanced(
            args.file,
            model_size=args.model,
            language=args.language,
            output_format=args.format,
            device=args.device,
            detect_language=args.detect_language,
            quality_mode=args.quality,
            generate_summary=args.summarize,
            summary_types=args.summary_types
        )
        
        console.print(f"\n✅ [bold green]Processamento concluído com sucesso![/bold green]")
        console.print(f"📄 Transcrição salva em: [bold]{output_path}[/bold]")
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Processo interrompido pelo usuário[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Erro: {e}[/bold red]")
        import traceback
        if "--debug" in sys.argv:
            console.print("\n[dim]Traceback completo:[/dim]")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()