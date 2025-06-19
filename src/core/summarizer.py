"""
Módulo de sumarização usando modelos de linguagem locais (Ollama).
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from enum import Enum
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.table import Table

console = Console()


class SummaryType(Enum):
    """Tipos de resumo disponíveis."""
    EXECUTIVE = "executive"          # Resumo executivo conciso
    TECHNICAL = "technical"          # Resumo técnico detalhado
    BULLET_POINTS = "bullet_points"  # Lista de pontos principais
    ACADEMIC = "academic"            # Resumo acadêmico formal
    SIMPLIFIED = "simplified"        # Versão simplificada
    COMPREHENSIVE = "comprehensive"  # Análise completa


class OllamaSummarizer:
    """Sistema de sumarização usando Ollama."""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa o sumarizador.
        
        Args:
            model: Modelo Ollama a usar (llama3.2, mistral, etc)
        """
        self.model = model
        self.ollama_url = "http://localhost:11434"
        self._check_ollama()
        
    def _check_ollama(self):
        """Verifica se o Ollama está instalado e rodando."""
        try:
            # Verificar se o Ollama está instalado
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError("Ollama não está instalado")
            
            # Verificar se está rodando
            import requests
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                if response.status_code != 200:
                    self._start_ollama()
            except:
                self._start_ollama()
                
            # Verificar se o modelo está disponível
            self._ensure_model()
            
        except FileNotFoundError:
            console.print("[bold red]❌ Ollama não encontrado![/bold red]")
            console.print("Instale o Ollama: https://ollama.ai/download")
            console.print("\nApós instalar, execute:")
            console.print("  ollama pull llama3.2:3b")
            raise RuntimeError("Ollama é necessário para resumos")
    
    def _start_ollama(self):
        """Inicia o servidor Ollama."""
        console.print("🚀 Iniciando servidor Ollama...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)  # Aguardar inicialização
    
    def _ensure_model(self):
        """Garante que o modelo está baixado."""
        import requests
        
        console.print(f"🔍 Verificando modelo {self.model}...")
        
        try:
            # Listar modelos disponíveis
            response = requests.get(f"{self.ollama_url}/api/tags")
            models = response.json().get("models", [])
            
            model_exists = any(m["name"] == self.model for m in models)
            
            if not model_exists:
                console.print(f"📥 Baixando modelo {self.model}...")
                subprocess.run(
                    ["ollama", "pull", self.model],
                    check=True
                )
                console.print(f"✅ Modelo {self.model} baixado com sucesso!")
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Erro ao verificar modelo: {e}[/yellow]")
    
    def summarize(
        self,
        text: str,
        summary_type: SummaryType = SummaryType.EXECUTIVE,
        max_length: Optional[int] = None,
        language: str = "pt",
        context: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, any]:
        """
        Gera um resumo do texto.
        
        Args:
            text: Texto para resumir
            summary_type: Tipo de resumo
            max_length: Tamanho máximo do resumo (palavras)
            language: Idioma do resumo
            context: Contexto adicional
            temperature: Criatividade (0-1)
            
        Returns:
            Dict com resumo e metadados
        """
        if not text or len(text.strip()) < 50:
            return {
                "summary": "Texto muito curto para resumir.",
                "type": summary_type.value,
                "error": "insufficient_content"
            }
        
        # Criar prompt baseado no tipo
        prompt = self._create_prompt(
            text, 
            summary_type, 
            max_length, 
            language, 
            context
        )
        
        # Gerar resumo
        console.print(f"\n🤖 Gerando {summary_type.value} com {self.model}...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Processando com IA...", total=None)
            
            start_time = time.time()
            
            try:
                import requests
                
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "stream": False,
                        "options": {
                            "num_predict": max_length * 2 if max_length else 1000,
                            "stop": ["</summary>", "---", "###"]
                        }
                    },
                    timeout=120
                )
                
                response.raise_for_status()
                result = response.json()
                
                elapsed = time.time() - start_time
                
                return {
                    "summary": result["response"].strip(),
                    "type": summary_type.value,
                    "model": self.model,
                    "processing_time": elapsed,
                    "tokens": result.get("eval_count", 0),
                    "language": language
                }
                
            except Exception as e:
                console.print(f"[red]❌ Erro ao gerar resumo: {e}[/red]")
                return {
                    "summary": f"Erro ao processar: {str(e)}",
                    "type": summary_type.value,
                    "error": str(e)
                }
    
    def _create_prompt(
        self,
        text: str,
        summary_type: SummaryType,
        max_length: Optional[int],
        language: str,
        context: Optional[str]
    ) -> str:
        """
        Cria o prompt apropriado para cada tipo de resumo.
        
        Args:
            text: Texto para resumir
            summary_type: Tipo de resumo
            max_length: Tamanho máximo
            language: Idioma
            context: Contexto adicional
            
        Returns:
            Prompt formatado
        """
        lang_instruction = {
            "pt": "Responda SEMPRE em português brasileiro.",
            "en": "Respond in English.",
            "es": "Responda en español."
        }.get(language, "Responda em português brasileiro.")
        
        length_instruction = f"O resumo deve ter no máximo {max_length} palavras." if max_length else ""
        context_instruction = f"Contexto adicional: {context}" if context else ""
        
        prompts = {
            SummaryType.EXECUTIVE: f"""
{lang_instruction}
Você é um assistente especializado em criar resumos executivos concisos e profissionais.

{context_instruction}

Crie um resumo executivo do seguinte texto, destacando os pontos mais importantes de forma clara e direta.
{length_instruction}

Formato do resumo:
- Parágrafo único ou dois parágrafos no máximo
- Linguagem profissional e objetiva
- Foco nos resultados e decisões principais

Texto para resumir:
{text}

Resumo executivo:
""",
            
            SummaryType.TECHNICAL: f"""
{lang_instruction}
Você é um especialista técnico criando documentação detalhada.

{context_instruction}

Analise o texto a seguir e crie um resumo técnico completo, incluindo:
1. Conceitos técnicos mencionados
2. Tecnologias e ferramentas
3. Processos e metodologias
4. Problemas e soluções
{length_instruction}

Texto para análise:
{text}

Resumo técnico:
""",
            
            SummaryType.BULLET_POINTS: f"""
{lang_instruction}
Crie uma lista com os pontos principais do texto.

{context_instruction}

Formato:
• Use bullet points (•)
• Cada ponto deve ser conciso e claro
• Ordene por importância
• {length_instruction}

Texto:
{text}

Pontos principais:
""",
            
            SummaryType.ACADEMIC: f"""
{lang_instruction}
Você é um pesquisador acadêmico criando um resumo formal.

{context_instruction}

Crie um resumo acadêmico seguindo esta estrutura:
1. Introdução/Contexto
2. Pontos principais
3. Metodologia (se aplicável)
4. Conclusões
{length_instruction}

Use linguagem formal e cite conceitos importantes.

Texto:
{text}

Resumo acadêmico:
""",
            
            SummaryType.SIMPLIFIED: f"""
{lang_instruction}
Explique o conteúdo de forma muito simples, como se estivesse explicando para alguém sem conhecimento técnico.

{context_instruction}

Use:
- Linguagem simples e clara
- Analogias quando apropriado
- Evite jargões técnicos
{length_instruction}

Texto:
{text}

Explicação simplificada:
""",
            
            SummaryType.COMPREHENSIVE: f"""
{lang_instruction}
Faça uma análise completa e detalhada do texto.

{context_instruction}

Inclua:
1. Resumo geral
2. Tópicos principais discutidos
3. Detalhes importantes
4. Contexto e implicações
5. Conclusões e insights
{length_instruction}

Texto para análise:
{text}

Análise completa:
"""
        }
        
        return prompts.get(summary_type, prompts[SummaryType.EXECUTIVE])
    
    def summarize_transcription(
        self,
        transcription_result: Dict,
        summary_types: List[SummaryType] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Gera múltiplos tipos de resumo para uma transcrição.
        
        Args:
            transcription_result: Resultado da transcrição
            summary_types: Lista de tipos de resumo desejados
            **kwargs: Argumentos adicionais para summarize()
            
        Returns:
            Dict com todos os resumos gerados
        """
        if summary_types is None:
            summary_types = [
                SummaryType.EXECUTIVE,
                SummaryType.BULLET_POINTS,
                SummaryType.TECHNICAL
            ]
        
        text = transcription_result.get("text", "")
        metadata = transcription_result.get("metadata", {})
        
        # Adicionar contexto da transcrição
        context = f"""
Arquivo: {metadata.get('file', 'N/A')}
Duração: {metadata.get('duration', 0):.1f} segundos
Idioma: {metadata.get('language', 'pt')}
"""
        
        summaries = {}
        
        for summary_type in summary_types:
            console.print(f"\n📝 Gerando {summary_type.value}...")
            
            summary = self.summarize(
                text,
                summary_type=summary_type,
                context=context,
                language=metadata.get('language', 'pt'),
                **kwargs
            )
            
            summaries[summary_type.value] = summary
        
        return {
            "transcription_file": metadata.get('file'),
            "summaries": summaries,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def generate_insights(
        self,
        text: str,
        domain: str = "general",
        language: str = "pt"
    ) -> Dict[str, any]:
        """
        Gera insights e análises aprofundadas do texto.
        
        Args:
            text: Texto para análise
            domain: Domínio do conteúdo (tech, business, academic, etc)
            language: Idioma
            
        Returns:
            Dict com insights gerados
        """
        prompt = f"""
{{"pt": "Responda em português brasileiro.", "en": "Respond in English."}}.get(language, "Responda em português.")

Você é um analista especializado em {domain}.

Analise o seguinte texto e forneça:

1. **Temas Principais**: Identifique os 3-5 temas centrais
2. **Insights Chave**: Observações importantes não explicitamente mencionadas
3. **Padrões**: Padrões ou tendências identificadas
4. **Questões**: Perguntas importantes levantadas ou não respondidas
5. **Recomendações**: Sugestões baseadas na análise

Texto:
{text}

Análise:
"""
        
        console.print("\n🔍 Gerando insights aprofundados...")
        
        result = self.summarize(
            text="",  # Não usado, prompt já contém o texto
            summary_type=SummaryType.COMPREHENSIVE,
            temperature=0.7  # Mais criatividade para insights
        )
        
        # Modificar o prompt diretamente
        import requests
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": 0.7,
                    "stream": False
                },
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            return {
                "insights": result["response"].strip(),
                "domain": domain,
                "model": self.model,
                "language": language
            }
            
        except Exception as e:
            console.print(f"[red]❌ Erro ao gerar insights: {e}[/red]")
            return {
                "insights": f"Erro ao processar: {str(e)}",
                "error": str(e)
            }
    
    def create_structured_summary(
        self,
        text: str,
        sections: List[str] = None,
        language: str = "pt"
    ) -> Dict[str, any]:
        """
        Cria um resumo estruturado com seções específicas.
        
        Args:
            text: Texto para resumir
            sections: Seções desejadas
            language: Idioma
            
        Returns:
            Dict com resumo estruturado
        """
        if sections is None:
            sections = [
                "Introdução",
                "Pontos Principais",
                "Detalhes Técnicos",
                "Conclusões",
                "Próximos Passos"
            ]
        
        sections_str = "\n".join(f"## {s}" for s in sections)
        
        prompt = f"""
{{"pt": "Responda em português brasileiro.", "en": "Respond in English."}}.get(language, "Responda em português.")

Crie um resumo estruturado do texto seguindo exatamente estas seções:

{sections_str}

Para cada seção, forneça informações relevantes encontradas no texto.
Se uma seção não tiver informações aplicáveis, escreva "Não mencionado no texto."

Texto:
{text}

Resumo estruturado:
"""
        
        console.print("\n📋 Criando resumo estruturado...")
        
        import requests
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": 0.3,
                    "stream": False
                },
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            return {
                "structured_summary": result["response"].strip(),
                "sections": sections,
                "model": self.model,
                "language": language
            }
            
        except Exception as e:
            console.print(f"[red]❌ Erro ao criar resumo estruturado: {e}[/red]")
            return {
                "structured_summary": f"Erro ao processar: {str(e)}",
                "error": str(e)
            }
    
    def display_summary_results(self, results: Dict) -> None:
        """
        Exibe os resultados dos resumos de forma formatada.
        
        Args:
            results: Resultados dos resumos
        """
        console.print("\n" + "="*60)
        console.print("[bold cyan]RESUMOS GERADOS[/bold cyan]")
        console.print("="*60 + "\n")
        
        summaries = results.get("summaries", {})
        
        for summary_type, summary_data in summaries.items():
            if "error" in summary_data:
                console.print(f"\n[red]❌ {summary_type}:[/red]")
                console.print(f"   Erro: {summary_data['error']}")
                continue
            
            console.print(f"\n[bold green]📄 {summary_type.upper()}:[/bold green]")
            
            # Mostrar metadados
            if summary_data.get("processing_time"):
                console.print(f"   ⏱️  Tempo: {summary_data['processing_time']:.1f}s")
            if summary_data.get("tokens"):
                console.print(f"   📊 Tokens: {summary_data['tokens']}")
            
            # Mostrar resumo
            console.print()
            
            summary_text = summary_data.get("summary", "")
            
            # Formatar de acordo com o tipo
            if summary_type == "bullet_points":
                # Já está formatado como lista
                console.print(Panel(
                    summary_text,
                    border_style="green",
                    padding=(1, 2)
                ))
            elif summary_type in ["technical", "academic", "comprehensive"]:
                # Usar Markdown para melhor formatação
                console.print(Panel(
                    Markdown(summary_text),
                    border_style="blue",
                    padding=(1, 2)
                ))
            else:
                # Texto simples
                console.print(Panel(
                    summary_text,
                    border_style="cyan",
                    padding=(1, 2)
                ))
        
        console.print("\n" + "="*60)