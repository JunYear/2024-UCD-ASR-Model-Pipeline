import os
import whisper
import logging

def download_whisper_models(models, download_dir):
    """
    지정된 디렉토리에 Whisper 모델을 다운로드하는 함수

    Args:
        models (list): 다운로드할 Whisper 모델 이름 리스트.
        download_dir (str): 모델을 다운로드할 디렉토리 경로.
    """
    # 다운로드 디렉토리가 없으면 생성
    os.makedirs(download_dir, exist_ok=True)
    
    for model_name in models:
        logging.info(f"Loading model: {model_name}")
        # download_root 파라미터를 사용하여 모델 저장 디렉토리 지정
        model = whisper.load_model(model_name, download_root=download_dir)
        logging.info(f"Model {model_name} loaded and downloaded if not present.")

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    
    # Whisper 모델 리스트
    whisper_models = ['turbo']
    
    # 다운로드할 디렉토리 지정
    download_directory = 'C:/Users/yyt11/OneDrive/바탕 화면/{Github}/2024-UCD-ASR-Model-Pipeline/whipser_models'
    
    download_whisper_models(whisper_models, download_directory)
