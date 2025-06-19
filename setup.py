#!/usr/bin/env python3
"""
Script de configuração inicial do Sistema de Transcrição.
"""
import os
import sys
import subprocess
from pathlib import Path
import shutil

def check_python_version():
    """Verifica a versão do Python."""
    print("🔍 Verificando versão do Python...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 ou superior é necessário!")
        print(f"   Versão atual: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detectado")
    return True

def check_ffmpeg():
    """Verifica se o FFmpeg está instalado."""
    print("\n🔍 Verificando FFmpeg...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            print("✅ FFmpeg instalado")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ FFmpeg não encontrado!")
    print("   Por favor, instale o FFmpeg:")
    print("   - Ubuntu/Debian: sudo apt install ffmpeg")
    print("   - macOS: brew install ffmpeg")
    print("   - Windows: baixe de https://ffmpeg.org/download.html")
    return False

def create_directories():
    """Cria os diretórios necessários."""
    print("\n📁 Criando estrutura de diretórios...")
    
    directories = [
        "data/uploads",
        "data/transcripts",
        "data/temp",
        "logs",
        "tests/fixtures"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Criar arquivo .gitkeep para manter diretórios vazios no git
        gitkeep = Path(dir_path) / ".gitkeep"
        gitkeep.touch()
    
    print("✅ Diretórios criados")

def setup_environment():
    """Configura o arquivo .env."""
    print("\n⚙️  Configurando ambiente...")
    
    if not Path(".env").exists() and Path(".env.example").exists():
        shutil.copy(".env.example", ".env")
        print("✅ Arquivo .env criado a partir de .env.example")
    else:
        print("ℹ️  Arquivo .env já existe ou .env.example não encontrado")

def install_dependencies():
    """Instala as dependências do projeto."""
    print("\n📦 Instalando dependências...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements.txt", "--break-system-packages"
        ])
        print("✅ Dependências instaladas com sucesso")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao instalar dependências")
        return False

def download_sample_model():
    """Baixa o modelo tiny para teste inicial."""
    print("\n🤖 Preparando modelo Whisper...")
    print("   O modelo será baixado automaticamente no primeiro uso")
    print("   Modelos disponíveis: tiny (39MB), base (74MB), small (244MB), medium (769MB), large (1.5GB)")
    
    # Opcionalmente, podemos pré-baixar o modelo tiny
    response = input("\n   Deseja baixar o modelo 'tiny' agora para teste? (s/N): ")
    if response.lower() == 's':
        try:
            import whisper
            print("   Baixando modelo tiny...")
            whisper.load_model("tiny")
            print("✅ Modelo tiny baixado com sucesso")
        except Exception as e:
            print(f"⚠️  Não foi possível baixar o modelo: {e}")

def create_sample_test():
    """Cria um arquivo de teste simples."""
    print("\n🧪 Criando arquivo de teste...")
    
    test_content = '''"""
Teste simples do sistema de transcrição.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_import():
    """Testa se os módulos podem ser importados."""
    try:
        from src.core.transcriber import WhisperTranscriber
        from src.core.audio_processor import AudioProcessor
        from src.core.config import WHISPER_CONFIG
        print("✅ Todos os módulos importados com sucesso!")
        return True
    except ImportError as e:
        print(f"❌ Erro ao importar módulos: {e}")
        return False

if __name__ == "__main__":
    test_import()
'''
    
    test_file = Path("tests/test_setup.py")
    test_file.write_text(test_content)
    print("✅ Arquivo de teste criado")

def main():
    """Função principal do setup."""
    print("🎙️  CONFIGURAÇÃO DO SISTEMA DE TRANSCRIÇÃO DE ÁUDIO")
    print("=" * 50)
    
    # Verificações
    if not check_python_version():
        return 1
    
    ffmpeg_ok = check_ffmpeg()
    
    # Configurações
    create_directories()
    setup_environment()
    
    # Instalação
    if not install_dependencies():
        return 1
    
    # Preparações finais
    download_sample_model()
    create_sample_test()
    
    # Resumo
    print("\n" + "=" * 50)
    print("📋 RESUMO DA INSTALAÇÃO")
    print("=" * 50)
    
    print(f"✅ Python: OK")
    print(f"{'✅' if ffmpeg_ok else '⚠️ '} FFmpeg: {'OK' if ffmpeg_ok else 'Necessário instalar'}")
    print(f"✅ Dependências: Instaladas")
    print(f"✅ Diretórios: Criados")
    print(f"✅ Ambiente: Configurado")
    
    print("\n🚀 PRÓXIMOS PASSOS:")
    print("1. Se o FFmpeg não está instalado, instale-o")
    print("2. Execute um teste: python main.py --list-models")
    print("3. Transcreva seu primeiro arquivo: python main.py arquivo.mp3")
    
    print("\n📚 Para mais informações, consulte o README.md")
    print("\n✨ Configuração concluída com sucesso!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())