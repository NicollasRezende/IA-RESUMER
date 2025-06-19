"""
Testes unitários para o módulo de transcrição.
"""
import pytest
import tempfile
from pathlib import Path
import json
import sys

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.transcriber import WhisperTranscriber
from src.core.config import TRANSCRIPT_DIR


class TestWhisperTranscriber:
    """Testes para a classe WhisperTranscriber."""
    
    @pytest.fixture
    def transcriber(self):
        """Cria uma instância do transcritor para testes."""
        # Usar modelo tiny para testes rápidos
        return WhisperTranscriber(model_size="tiny", device="cpu")
    
    @pytest.fixture
    def sample_audio_path(self):
        """Retorna o caminho para um arquivo de áudio de teste."""
        # Criar um arquivo de áudio sintético para testes
        # Em produção, você deve ter um arquivo real em tests/fixtures/
        test_file = Path(__file__).parent / "fixtures" / "sample_audio.wav"
        
        # Se o arquivo não existir, pular o teste
        if not test_file.exists():
            pytest.skip("Arquivo de teste não encontrado")
        
        return test_file
    
    def test_initialization(self, transcriber):
        """Testa a inicialização do transcritor."""
        assert transcriber is not None
        assert transcriber.model is not None
        assert transcriber.model_size == "tiny"
        assert transcriber.device in ["cpu", "cuda"]
    
    def test_detect_language(self, transcriber, sample_audio_path):
        """Testa a detecção de idioma."""
        # Este teste requer um arquivo de áudio real
        language = transcriber.detect_language(sample_audio_path)
        
        # Verificar se retornou um código de idioma válido
        assert isinstance(language, str)
        assert len(language) == 2  # Códigos de idioma têm 2 caracteres
    
    def test_transcribe_basic(self, transcriber, sample_audio_path):
        """Testa transcrição básica."""
        result = transcriber.transcribe(sample_audio_path)
        
        # Verificar estrutura do resultado
        assert "text" in result
        assert "segments" in result
        assert "metadata" in result
        
        # Verificar metadados
        metadata = result["metadata"]
        assert "file" in metadata
        assert "duration" in metadata
        assert "processing_time" in metadata
        assert "model" in metadata
        assert metadata["model"] == "tiny"
    
    def test_transcribe_with_segments(self, transcriber, sample_audio_path):
        """Testa transcrição com segmentos detalhados."""
        result = transcriber.transcribe_with_segments(sample_audio_path)
        
        assert "text" in result
        assert "segments" in result
        assert isinstance(result["segments"], list)
        
        # Verificar estrutura dos segmentos
        if result["segments"]:
            segment = result["segments"][0]
            assert "id" in segment
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment
            assert "duration" in segment
    
    def test_save_transcription_json(self, transcriber, sample_audio_path):
        """Testa salvamento em formato JSON."""
        result = transcriber.transcribe(sample_audio_path)
        
        # Salvar como JSON
        output_path = transcriber.save_transcription(result, format="json")
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
        
        # Verificar conteúdo
        with open(output_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        
        assert saved_data["text"] == result["text"]
        
        # Limpar
        output_path.unlink()
    
    def test_save_transcription_txt(self, transcriber, sample_audio_path):
        """Testa salvamento em formato TXT."""
        result = transcriber.transcribe(sample_audio_path)
        
        # Salvar como TXT
        output_path = transcriber.save_transcription(result, format="txt")
        
        assert output_path.exists()
        assert output_path.suffix == ".txt"
        
        # Verificar conteúdo
        with open(output_path, "r", encoding="utf-8") as f:
            saved_text = f.read()
        
        assert saved_text == result["text"]
        
        # Limpar
        output_path.unlink()
    
    def test_save_transcription_srt(self, transcriber, sample_audio_path):
        """Testa salvamento em formato SRT."""
        result = transcriber.transcribe_with_segments(sample_audio_path)
        
        # Salvar como SRT
        output_path = transcriber.save_transcription(result, format="srt")
        
        assert output_path.exists()
        assert output_path.suffix == ".srt"
        
        # Verificar formato básico SRT
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # SRT deve ter números de sequência e timestamps
        assert "1" in content
        assert "-->" in content
        
        # Limpar
        output_path.unlink()
    
    def test_get_device_info(self, transcriber):
        """Testa obtenção de informações do dispositivo."""
        info = transcriber.get_device_info()
        
        assert "device" in info
        assert "device_name" in info
        assert "cuda_available" in info
        
        if info["device"] == "cuda":
            assert "gpu_memory_total" in info
            assert "gpu_memory_allocated" in info
    
    def test_seconds_to_time_conversion(self):
        """Testa conversão de segundos para formato de tempo."""
        # Testar conversão estática
        assert WhisperTranscriber._seconds_to_time(0) == "00:00:00,000"
        assert WhisperTranscriber._seconds_to_time(61.5) == "00:01:01,500"
        assert WhisperTranscriber._seconds_to_time(3661.5) == "01:01:01,500"
        
        # Testar formato VTT
        assert WhisperTranscriber._seconds_to_time(61.5, vtt=True) == "00:01:01.500"
    
    @pytest.mark.parametrize("model_size", ["tiny", "base", "small"])
    def test_different_models(self, model_size):
        """Testa inicialização com diferentes tamanhos de modelo."""
        transcriber = WhisperTranscriber(model_size=model_size, device="cpu")
        assert transcriber.model_size == model_size
        assert transcriber.model is not None
    
    def test_invalid_audio_path(self, transcriber):
        """Testa comportamento com arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe("arquivo_inexistente.wav")
    
    def test_transcribe_with_options(self, transcriber, sample_audio_path):
        """Testa transcrição com opções customizadas."""
        result = transcriber.transcribe(
            sample_audio_path,
            language="pt",
            temperature=0.5,
            initial_prompt="Este é um teste de transcrição."
        )
        
        assert result is not None
        assert result["metadata"]["language"] == "pt"