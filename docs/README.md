# 🎙️ Sistema Avançado de Transcrição e Resumo de Áudio/Vídeo v2.0

Sistema completo, robusto e gratuito para transcrição de alta qualidade e geração de resumos inteligentes usando OpenAI Whisper e Ollama.

## 🚀 Novidades da v2.0

- **🎯 Transcrição Aprimorada**: Melhor qualidade com detecção de problemas e auto-correção
- **🤖 Resumos com IA**: 6 tipos diferentes de resumo usando modelos locais
- **📊 Relatórios de Qualidade**: Métricas detalhadas sobre a transcrição
- **💾 Sistema de Cache**: Evita reprocessamento desnecessário
- **🔄 Processamento Robusto**: Suporte melhorado para arquivos longos
- **🎨 Interface Interativa**: Menu completo com todas as funcionalidades

## ✨ Características Principais

### Transcrição
- 🎯 **Qualidade Superior** com modo enhanced
- 🌍 **Detecção Automática de Idioma**
- 📁 **Múltiplos Formatos**: MP3, WAV, MP4, MKV, AVI, FLAC, OGG, e mais
- 📊 **Múltiplos Outputs**: JSON, TXT, SRT, VTT
- 🚀 **Otimizado para GPU** com fallback para CPU
- 🔧 **Auto-seleção de Modelo** baseada na qualidade necessária

### Resumos com IA
- 📝 **6 Tipos de Resumo**:
  - Executive (Resumo Executivo)
  - Bullet Points (Pontos Principais)
  - Technical (Análise Técnica)
  - Simplified (Versão Simplificada)  
  - Comprehensive (Análise Completa)
  - Academic (Formato Acadêmico)
- 🔍 **Geração de Insights** automática
- 🌐 **Suporte Multilíngue**
- 💻 **100% Local** - sem envio de dados

### Robustez
- ✂️ **Chunking Inteligente** para arquivos longos
- 🔄 **Retry Automático** com múltiplos modelos
- 📈 **Métricas de Qualidade** em tempo real
- 💾 **Cache Inteligente** de resultados
- 🧹 **Limpeza Automática** de arquivos temporários

## 📋 Requisitos

- Python 3.8+
- FFmpeg
- 16GB RAM (32GB recomendado para modelos grandes)
- GPU com CUDA (opcional, mas recomendado)
- ~10GB de espaço livre

## 🛠️ Instalação Completa

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/audio-transcription-system.git
cd audio-transcription-system
```

### 2. Instale o FFmpeg

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

### 3. Configure o ambiente Python

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 4. Instale o Ollama (para resumos)

```bash
# Linux/Mac
curl https://ollama.ai/install.sh | sh

# Windows
# Baixe de: https://ollama.ai/download

# Baixar modelo para resumos
ollama pull llama3.2:3b
```

### 5. Instale a biblioteca Magic (opcional, melhora validação)

```bash
# Ubuntu/Debian
sudo apt-get install libmagic1

# macOS
brew install libmagic

# Python
pip install python-magic
```

## 🚀 Uso Rápido

### Modo Interativo (Recomendado)

```bash
python main_enhanced.py -i
```

### Transcrição Básica

```bash
# Transcrição simples
python main_enhanced.py audio.mp3

# Com modelo específico
python main_enhanced.py video.mp4 -m medium

# Com detecção de idioma
python main_enhanced.py podcast.wav --detect-language

# Modo de qualidade aprimorada
python main_enhanced.py aula.mp4 -q enhanced
```

### Transcrição com Resumo Automático

```bash
# Gerar resumo executivo e pontos principais
python main_enhanced.py entrevista.mp3 -s

# Escolher tipos específicos de resumo
python main_enhanced.py palestra.mp4 -s --summary-types executive technical bullet_points

# Resumo completo com insights
python main_enhanced.py reuniao.wav -s --summary-types comprehensive
```

### Exemplos Avançados

```bash
# Auto-seleção de modelo baseada na qualidade
python main_enhanced.py importante.mp3 -m auto -q enhanced

# Processamento em lote (via script)
for file in *.mp3; do
    python main_enhanced.py "$file" -s --summary-types executive
done

# Verificar sistema antes de processar
python main_enhanced.py --check

# Listar transcrições recentes
python main_enhanced.py --list
```

## 📊 Modos de Qualidade

### Basic
- Rápido e eficiente
- Ideal para arquivos com áudio limpo
- Usa modelo único

### Enhanced (Padrão)
- Melhor qualidade de transcrição
- Pré-processamento de áudio
- Detecção e correção de problemas
- Chunking inteligente para arquivos longos

### Auto
- Escolhe automaticamente baseado no arquivo
- Enhanced para < 10 minutos
- Basic para arquivos maiores

## 🎯 Tipos de Resumo Disponíveis

### Executive Summary
```
Resumo conciso focado em decisões e resultados principais.
Ideal para: Relatórios executivos, briefings rápidos
```

### Bullet Points
```
• Pontos principais organizados
• Fácil leitura e escaneamento
• Informações mais importantes destacadas
Ideal para: Apresentações, revisões rápidas
```

### Technical Summary
```
Análise detalhada de aspectos técnicos, tecnologias mencionadas,
processos e metodologias discutidas.
Ideal para: Documentação técnica, análise de conteúdo especializado
```

### Simplified Version
```
Explicação em linguagem simples e acessível, removendo jargões
técnicos e usando analogias quando apropriado.
Ideal para: Audiência geral, material educacional
```

### Comprehensive Analysis
```
Análise completa incluindo contexto, implicações, detalhes
importantes e insights não óbvios.
Ideal para: Pesquisa, análise aprofundada
```

### Academic Format
```
Formato estruturado com introdução, metodologia (se aplicável),
pontos principais e conclusões.
Ideal para: Papers, trabalhos acadêmicos
```

## 📁 Estrutura de Saída

```
data/
├── uploads/          # Arquivos originais
├── transcripts/      # Transcrições em JSON/TXT/SRT
│   ├── audio_20240619_143022.json
│   ├── audio_20240619_143022_summaries.json
│   └── .cache_transcription_*.json
└── temp/            # Arquivos temporários (limpos automaticamente)
```

### Exemplo de Saída JSON

```json
{
  "text": "Transcrição completa...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Primeira frase transcrita.",
      "words": [...],
      "avg_confidence": 0.95
    }
  ],
  "metadata": {
    "file": "audio.mp3",
    "duration": 361.5,
    "processing_time": 45.3,
    "model": "medium",
    "language": "pt",
    "method": "chunks"
  },
  "quality_metrics": {
    "total_segments": 30,
    "avg_segment_confidence": 0.92,
    "low_confidence_segments": 2,
    "silence_ratio": 0.15
  }
}
```

### Exemplo de Resumo

```json
{
  "summaries": {
    "executive": {
      "summary": "Este tutorial demonstra a implementação de hooks...",
      "processing_time": 3.2,
      "model": "llama3.2:3b"
    },
    "bullet_points": {
      "summary": "• Conceito principal: uso de service hooks\n• Tecnologia: Liferay...",
      "processing_time": 2.8
    }
  }
}
```

## ⚙️ Configuração Avançada

### Variáveis de Ambiente (.env)

```env
# Modelos e Performance
WHISPER_MODEL=small
WHISPER_DEVICE=cuda
WHISPER_LANGUAGE=pt
BATCH_SIZE=4
NUM_WORKERS=8

# Cache
CACHE_ENABLED=True
CACHE_TTL=86400  # 24 horas

# Ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_URL=http://localhost:11434
```

### Configuração de Hardware

| Configuração | Mínimo | Recomendado | Ideal |
|-------------|---------|-------------|-------|
| CPU | i5/Ryzen 5 | i7/Ryzen 7 | i9/Ryzen 9 |
| RAM | 8GB | 16GB | 32GB+ |
| GPU | - | GTX 1660 (6GB) | RTX 4070+ (12GB+) |
| Armazenamento | 10GB | 50GB | 100GB+ SSD |

## 🔧 Solução de Problemas

### Transcrição Incompleta ("...")

```bash
# Use modelo maior
python main_enhanced.py arquivo.mp3 -m medium -q enhanced

# Ou modo auto para seleção inteligente
python main_enhanced.py arquivo.mp3 -m auto
```

### Erro de Memória

```bash
# Use modelo menor
python main_enhanced.py arquivo.mp3 -m tiny

# Ou processe em chunks menores
# Edite AUDIO_CONFIG["chunk_duration"] para 120 (2 minutos)
```

### Ollama não Funciona

```bash
# Verificar se está rodando
ollama list

# Iniciar manualmente
ollama serve

# Baixar modelo
ollama pull llama3.2:3b
```

### Cache Corrompido

```bash
# Limpar cache manualmente
rm data/transcripts/.cache_*

# Ou via interface
python main_enhanced.py -i
# Escolha opção 5 (Limpar arquivos)
```

## 📈 Performance e Benchmarks

### Velocidade de Transcrição (RTX 4070)

| Modelo | Velocidade | Qualidade | RAM | VRAM |
|--------|------------|-----------|-----|------|
| Tiny | ~50x | ★★☆☆☆ | 4GB | 1GB |
| Small | ~25x | ★★★★☆ | 8GB | 2GB |
| Medium | ~10x | ★★★★★ | 16GB | 5GB |
| Large | ~5x | ★★★★★ | 24GB | 10GB |

### Tempo de Processamento (arquivo de 1 hora)

| Processo | CPU Only | Com GPU |
|----------|----------|---------|
| Transcrição (small) | ~15 min | ~2.5 min |
| Transcrição (medium) | ~30 min | ~6 min |
| Resumo (todos os tipos) | ~2 min | ~2 min |
| Total com cache | Instantâneo | Instantâneo |

## 🗺️ Roadmap

- [x] Transcrição básica com Whisper
- [x] Suporte a múltiplos formatos
- [x] Sistema de resumo com IA
- [x] Interface interativa
- [x] Cache inteligente
- [x] Métricas de qualidade
- [ ] Interface Web (Gradio/Streamlit)
- [ ] API REST completa
- [ ] Suporte para streaming ao vivo
- [ ] Tradução automática
- [ ] Integração com serviços de nuvem
- [ ] App desktop (Electron)

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

## 📞 Suporte e Comunidade

- 📧 Email: seu-email@exemplo.com
- 💬 Discord: [Link do servidor]
- 📚 Wiki: [GitHub Wiki](https://github.com/seu-usuario/audio-transcription-system/wiki)
- 🐛 Issues: [GitHub Issues](https://github.com/seu-usuario/audio-transcription-system/issues)

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- [OpenAI Whisper](https://github.com/openai/whisper) - Tecnologia de transcrição
- [Ollama](https://ollama.ai) - LLMs locais para resumos
- [FFmpeg](https://ffmpeg.org/) - Processamento de mídia
- Comunidade open source pelos modelos e ferramentas

---

**Feito com ❤️ e 100% tecnologias open source**

*"Transformando áudio em conhecimento estruturado"*