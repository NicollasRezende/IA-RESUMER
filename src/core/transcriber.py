"""
Módulo de transcrição aprimorado com melhor qualidade e robustez.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings
import hashlib

import whisper
import torch
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table

from .config import (
    WHISPER_CONFIG, 
    TRANSCRIPT_DIR, 
    get_model_info,
    OUTPUT_CONFIG,
    TEMP_DIR
)
from .audio_processor import AudioProcessor

warnings.filterwarnings("ignore", category=UserWarning)
console = Console()


class WhisperTranscriber:
    """Transcritor aprimorado com melhor qualidade e features avançadas."""
    
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None):
        """
        Inicializa o transcritor aprimorado.
        
        Args:
            model_size: Tamanho do modelo (tiny, base, small, medium, large)
            device: Dispositivo para execução (cpu, cuda)
        """
        self.model_size = model_size or WHISPER_CONFIG["model_size"]
        self.device = device or WHISPER_CONFIG["device"]
        self.model = None
        self.audio_processor = AudioProcessor()
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
                download_root=None
            )
            
            elapsed = time.time() - start_time
            progress.update(task, completed=True)
            
        console.print(f"✅ Modelo carregado em {elapsed:.1f}s no dispositivo: [bold green]{self.device}[/bold green]\n")
    
    def transcribe_enhanced(
        self, 
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        temperature: Union[float, List[float]] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold: float = 2.4,
        logprob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = True,
        **kwargs
    ) -> Dict:
        """
        Transcrição aprimorada com melhor qualidade e robustez.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            language: Idioma (None para detecção automática)
            task: 'transcribe' ou 'translate'
            initial_prompt: Prompt para guiar o modelo
            temperature: Temperatura(s) para fallback em caso de falha
            compression_ratio_threshold: Threshold para detectar repetições
            logprob_threshold: Threshold para detectar baixa confiança
            no_speech_threshold: Threshold para detectar silêncio
            condition_on_previous_text: Usar contexto anterior
            
        Returns:
            Dict com transcrição completa e metadados
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        
        console.print(f"🎵 Processando: [bold]{audio_path.name}[/bold]")
        
        # Pré-processar áudio se necessário
        processed_path = self._preprocess_audio(audio_path)
        
        # Detectar idioma se não especificado
        if not language:
            language = self.detect_language(processed_path)
        
        # Configurar parâmetros otimizados
        params = {
            "language": language,
            "task": task,
            "fp16": WHISPER_CONFIG["fp16"] and self.device == "cuda",
            "verbose": False,  # Desabilitar verbose padrão
            "temperature": temperature,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "word_timestamps": True,  # Sempre usar para melhor precisão
            "initial_prompt": initial_prompt or self._get_language_prompt(language),
            "beam_size": 5,  # Melhor qualidade
            "best_of": 5,   # Múltiplas tentativas
            "patience": 1.0,
        }
        
        # Atualizar com kwargs
        params.update(kwargs)
        
        # Transcrever com estratégia de chunks para arquivos longos
        audio_info = self.audio_processor.get_audio_info(processed_path)
        duration = audio_info.get("duration", 0)
        
        if duration > 300:  # Se maior que 5 minutos, usar chunks
            console.print(f"📊 Áudio longo detectado ({duration:.1f}s), usando processamento em chunks...")
            result = self._transcribe_long_audio(processed_path, params)
        else:
            result = self._transcribe_single(processed_path, params)
        
        # Pós-processar resultado
        result = self._postprocess_transcription(result)
        
        # Limpar arquivo temporário se criado
        if processed_path != audio_path and processed_path.exists():
            processed_path.unlink()
        
        return result
    
    def _preprocess_audio(self, audio_path: Path) -> Path:
        """
        Pré-processa o áudio para melhor qualidade de transcrição.
        
        Args:
            audio_path: Caminho do áudio original
            
        Returns:
            Path do áudio processado
        """
        # Verificar se precisa normalização
        audio_info = self.audio_processor.get_audio_info(audio_path)
        
        # Se já está no formato ideal, retornar
        if (audio_info.get("sample_rate") == 16000 and 
            audio_info.get("channels") == 1 and
            audio_path.suffix.lower() == '.wav'):
            return audio_path
        
        console.print("🔧 Otimizando áudio para transcrição...")
        
        # Converter e normalizar
        processed_path = self.audio_processor.convert_to_wav(audio_path)
        normalized_path = self.audio_processor.normalize_audio(processed_path)
        
        # Remover arquivo intermediário
        if processed_path != audio_path and processed_path.exists():
            processed_path.unlink()
        
        return normalized_path
    
    def _transcribe_single(self, audio_path: Path, params: Dict) -> Dict:
        """
        Transcreve um único arquivo de áudio.
        
        Args:
            audio_path: Caminho do áudio
            params: Parâmetros para transcrição
            
        Returns:
            Resultado da transcrição
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Transcrevendo áudio...", total=100)
            
            start_time = time.time()
            
            # Callback para progresso
            def callback(current, total):
                progress.update(task, completed=int((current / total) * 100))
            
            # Transcrever com Whisper
            result = self.model.transcribe(
                str(audio_path),
                **params,
                progress_callback=callback if hasattr(self.model, 'transcribe_with_progress') else None
            )
            
            progress.update(task, completed=100)
            elapsed = time.time() - start_time
        
        # Adicionar metadados
        result["processing_time"] = elapsed
        result["method"] = "single"
        
        return result
    
    def _transcribe_long_audio(self, audio_path: Path, params: Dict) -> Dict:
        """
        Transcreve áudio longo usando estratégia de chunks.
        
        Args:
            audio_path: Caminho do áudio
            params: Parâmetros para transcrição
            
        Returns:
            Resultado combinado da transcrição
        """
        # Dividir áudio em chunks por silêncio ou duração
        console.print("✂️  Dividindo áudio em segmentos...")
        
        # Tentar dividir por silêncio primeiro
        chunks = self.audio_processor.split_audio_by_silence(
            audio_path,
            min_silence_len=500,    # 500ms de silêncio
            silence_thresh=-35,     # Threshold mais sensível
            keep_silence=250        # Manter 250ms de contexto
        )
        
        # Se não encontrou silêncios suficientes, dividir por tempo
        if len(chunks) < 2:
            chunks = self.audio_processor.split_audio_by_duration(
                audio_path,
                chunk_duration=240  # 4 minutos por chunk
            )
        
        console.print(f"📚 Processando {len(chunks)} segmentos...")
        
        # Transcrever cada chunk
        all_segments = []
        full_text = []
        total_duration = 0
        
        for i, chunk_path in enumerate(chunks):
            console.print(f"\n🎤 Segmento {i+1}/{len(chunks)}")
            
            # Usar prompt do segmento anterior para continuidade
            if i > 0 and all_segments:
                # Pegar últimas palavras do segmento anterior
                prev_text = all_segments[-1]["text"]
                last_words = " ".join(prev_text.split()[-10:])
                params["initial_prompt"] = f"...{last_words}"
            
            # Transcrever chunk
            chunk_result = self._transcribe_single(chunk_path, params.copy())
            
            # Ajustar timestamps
            if all_segments:
                time_offset = all_segments[-1]["end"]
                for segment in chunk_result.get("segments", []):
                    segment["start"] += time_offset
                    segment["end"] += time_offset
                    segment["id"] = len(all_segments) + segment["id"]
            
            # Adicionar aos resultados
            all_segments.extend(chunk_result.get("segments", []))
            full_text.append(chunk_result.get("text", "").strip())
            
            # Atualizar duração
            if chunk_result.get("segments"):
                total_duration = chunk_result["segments"][-1]["end"]
            
            # Limpar chunk temporário
            chunk_path.unlink()
        
        # Combinar resultados
        result = {
            "text": " ".join(full_text),
            "segments": all_segments,
            "language": params["language"],
            "duration": total_duration,
            "method": "chunks",
            "num_chunks": len(chunks)
        }
        
        return result
    
    def _postprocess_transcription(self, result: Dict) -> Dict:
        """
        Pós-processa a transcrição para melhorar qualidade.
        
        Args:
            result: Resultado bruto da transcrição
            
        Returns:
            Resultado processado
        """
        console.print("\n🔧 Aplicando pós-processamento...")
        
        # Remover repetições excessivas
        result["text"] = self._remove_repetitions(result["text"])
        
        # Limpar segmentos
        cleaned_segments = []
        for segment in result.get("segments", []):
            # Remover segmentos vazios ou muito curtos
            if len(segment["text"].strip()) < 3:
                continue
            
            # Limpar texto do segmento
            segment["text"] = self._clean_segment_text(segment["text"])
            
            # Calcular confiança média se disponível
            if "words" in segment:
                confidences = [w.get("probability", 1.0) for w in segment["words"]]
                segment["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 1.0
            
            cleaned_segments.append(segment)
        
        result["segments"] = cleaned_segments
        
        # Adicionar estatísticas de qualidade
        result["quality_metrics"] = self._calculate_quality_metrics(result)
        
        return result
    
    def _remove_repetitions(self, text: str) -> str:
        """Remove repetições excessivas do texto."""
        import re
        
        # Remover múltiplos espaços
        text = re.sub(r'\s+', ' ', text)
        
        # Remover repetições de palavras (3+ vezes)
        text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text)
        
        # Remover múltiplos pontos consecutivos (mais de 3)
        text = re.sub(r'\.{4,}', '...', text)
        
        return text.strip()
    
    def _clean_segment_text(self, text: str) -> str:
        """Limpa o texto de um segmento."""
        # Remover espaços extras
        text = " ".join(text.split())
        
        # Capitalizar primeira letra
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Adicionar pontuação final se não houver
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _calculate_quality_metrics(self, result: Dict) -> Dict:
        """
        Calcula métricas de qualidade da transcrição.
        
        Args:
            result: Resultado da transcrição
            
        Returns:
            Dict com métricas de qualidade
        """
        segments = result.get("segments", [])
        
        metrics = {
            "total_segments": len(segments),
            "total_words": len(result["text"].split()),
            "avg_segment_confidence": 0,
            "low_confidence_segments": 0,
            "silence_ratio": 0,
        }
        
        if segments:
            # Calcular confiança média
            confidences = [s.get("avg_confidence", 1.0) for s in segments]
            metrics["avg_segment_confidence"] = sum(confidences) / len(confidences)
            
            # Contar segmentos de baixa confiança
            metrics["low_confidence_segments"] = sum(1 for c in confidences if c < 0.8)
            
            # Calcular proporção de silêncio
            total_duration = result.get("duration", 0)
            speech_duration = sum(s["end"] - s["start"] for s in segments)
            if total_duration > 0:
                metrics["silence_ratio"] = 1 - (speech_duration / total_duration)
        
        return metrics
    
    def _get_language_prompt(self, language: str) -> str:
        """
        Retorna um prompt apropriado para o idioma.
        
        Args:
            language: Código do idioma
            
        Returns:
            Prompt inicial para melhorar a transcrição
        """
        prompts = {
            "pt": "Esta é uma transcrição em português. O áudio pode conter termos técnicos.",
            "en": "This is an English transcription. The audio may contain technical terms.",
            "es": "Esta es una transcripción en español. El audio puede contener términos técnicos.",
        }
        
        return prompts.get(language, "")
    
    def transcribe_with_fallback(
        self,
        audio_path: Union[str, Path],
        models: List[str] = ["small", "medium", "large"],
        **kwargs
    ) -> Dict:
        """
        Tenta transcrever com múltiplos modelos em caso de falha.
        
        Args:
            audio_path: Caminho do áudio
            models: Lista de modelos para tentar em ordem
            **kwargs: Argumentos para transcrição
            
        Returns:
            Melhor resultado obtido
        """
        audio_path = Path(audio_path)
        best_result = None
        best_score = 0
        
        for model_size in models:
            try:
                console.print(f"\n🔄 Tentando com modelo [bold]{model_size}[/bold]...")
                
                # Criar novo transcritor com o modelo
                transcriber = WhisperTranscriber(
                    model_size=model_size,
                    device=self.device
                )
                
                # Transcrever
                result = transcriber.transcribe_enhanced(audio_path, **kwargs)
                
                # Calcular score de qualidade
                metrics = result.get("quality_metrics", {})
                score = (
                    metrics.get("avg_segment_confidence", 0) * 0.5 +
                    (1 - metrics.get("silence_ratio", 1)) * 0.3 +
                    (1 - metrics.get("low_confidence_segments", 0) / max(metrics.get("total_segments", 1), 1)) * 0.2
                )
                
                console.print(f"   Score de qualidade: [bold]{score:.2f}[/bold]")
                
                # Manter melhor resultado
                if score > best_score:
                    best_result = result
                    best_score = score
                    best_result["model_used"] = model_size
                
                # Se score é bom o suficiente, parar
                if score > 0.85:
                    break
                    
            except Exception as e:
                console.print(f"   [yellow]⚠️  Falha com {model_size}: {e}[/yellow]")
                continue
        
        if not best_result:
            raise RuntimeError("Falha ao transcrever com todos os modelos")
        
        console.print(f"\n✅ Melhor resultado com modelo: [bold green]{best_result['model_used']}[/bold green]")
        return best_result
    
    def generate_quality_report(self, result: Dict) -> None:
        """
        Gera um relatório de qualidade da transcrição.
        
        Args:
            result: Resultado da transcrição
        """
        console.print("\n" + "="*50)
        console.print("[bold cyan]RELATÓRIO DE QUALIDADE DA TRANSCRIÇÃO[/bold cyan]")
        console.print("="*50 + "\n")
        
        # Informações gerais
        metadata = result.get("metadata", {})
        metrics = result.get("quality_metrics", {})
        
        info_table = Table(show_header=False, box=None)
        info_table.add_column("Métrica", style="cyan")
        info_table.add_column("Valor", style="white")
        
        info_table.add_row("Arquivo:", metadata.get("file", "N/A"))
        info_table.add_row("Duração:", f"{metadata.get('duration', 0):.1f}s")
        info_table.add_row("Modelo:", metadata.get("model", "N/A"))
        info_table.add_row("Método:", result.get("method", "N/A"))
        info_table.add_row("Idioma:", result.get("language", "N/A"))
        
        console.print(info_table)
        console.print()
        
        # Métricas de qualidade
        quality_table = Table(title="Métricas de Qualidade", show_lines=True)
        quality_table.add_column("Métrica", style="cyan")
        quality_table.add_column("Valor", style="white")
        quality_table.add_column("Status", style="white")
        
        # Confiança média
        avg_conf = metrics.get("avg_segment_confidence", 0)
        conf_status = "[green]Ótimo[/green]" if avg_conf > 0.9 else "[yellow]Bom[/yellow]" if avg_conf > 0.8 else "[red]Baixo[/red]"
        quality_table.add_row("Confiança Média", f"{avg_conf:.1%}", conf_status)
        
        # Segmentos de baixa confiança
        low_conf = metrics.get("low_confidence_segments", 0)
        total_segs = metrics.get("total_segments", 1)
        low_conf_ratio = low_conf / total_segs if total_segs > 0 else 0
        low_conf_status = "[green]Ótimo[/green]" if low_conf_ratio < 0.1 else "[yellow]Aceitável[/yellow]" if low_conf_ratio < 0.2 else "[red]Alto[/red]"
        quality_table.add_row("Segmentos Baixa Confiança", f"{low_conf}/{total_segs} ({low_conf_ratio:.1%})", low_conf_status)
        
        # Taxa de silêncio
        silence_ratio = metrics.get("silence_ratio", 0)
        silence_status = "[green]Normal[/green]" if silence_ratio < 0.3 else "[yellow]Alto[/yellow]" if silence_ratio < 0.5 else "[red]Muito Alto[/red]"
        quality_table.add_row("Taxa de Silêncio", f"{silence_ratio:.1%}", silence_status)
        
        # Total de palavras
        total_words = metrics.get("total_words", 0)
        quality_table.add_row("Total de Palavras", str(total_words), "[green]OK[/green]")
        
        console.print(quality_table)
        
        # Recomendações
        console.print("\n[bold]Recomendações:[/bold]")
        
        if avg_conf < 0.8:
            console.print("• [yellow]Considere usar um modelo maior para melhor qualidade[/yellow]")
        
        if low_conf_ratio > 0.2:
            console.print("• [yellow]Muitos segmentos com baixa confiança - verifique a qualidade do áudio[/yellow]")
        
        if silence_ratio > 0.5:
            console.print("• [yellow]Alto índice de silêncio - considere editar o áudio antes da transcrição[/yellow]")
        
        if total_words < 50:
            console.print("• [yellow]Poucas palavras detectadas - verifique se o áudio contém fala[/yellow]")
        
        console.print("\n" + "="*50)