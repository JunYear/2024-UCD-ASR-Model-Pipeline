import os
import re
from jiwer import wer
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging
import glob
import pandas as pd  # 데이터 저장을 위한 pandas 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def clean_transcription(text):
    """
    Transcription 텍스트를 정리하여 화자 라벨과 괄호 제거를 위한 함수

    Args:
        text (str): Transcription 텍스트.

    Returns:
        str: 정리된 텍스트.
    """
    # 'SPEAKER XX: (text)' 형식 제거
    cleaned_text = re.sub(r'SPEAKER \d{2}: \((.*?)\)', r'\1', text, flags=re.IGNORECASE)

    # 남아있는 화자 라벨 제거
    cleaned_text = re.sub(r'SPEAKER \d{2}:', '', cleaned_text, flags=re.IGNORECASE)

    # 괄호 제거
    cleaned_text = cleaned_text.replace('(', '').replace(')', '')

    # 불필요한 공백 제거
    cleaned_text = ' '.join(cleaned_text.split())

    return cleaned_text

def compute_metrics(reference_path, hypothesis_path):
    """
    Reference와 Hypothesis 텍스트 파일 간의 WER과 CER을 계산을 위한 함수

    Args:
        reference_path (str): Reference 텍스트 파일 경로.
        hypothesis_path (str): Hypothesis 텍스트 파일 경로.

    Returns:
        tuple: (WER, CER)
    """
    # Reference 읽기
    with open(reference_path, 'r', encoding='utf-8') as f:
        reference = f.read()

    # Hypothesis 읽기
    with open(hypothesis_path, 'r', encoding='utf-8') as f:
        hypothesis = f.read()

    # Hypothesis 정리
    hypothesis_clean = clean_transcription(hypothesis)

    # Reference 정리 (필요 시)
    reference_clean = reference.strip()  # Reference에 화자 라벨이 없는 경우

    # WER 계산
    wer_score = wer(reference_clean, hypothesis_clean)

    # CER 계산
    # CER을 계산하기 위해 공백을 제거하고 각 문자를 단어로 간주
    reference_chars = reference_clean.replace(' ', '')
    hypothesis_chars = hypothesis_clean.replace(' ', '')
    cer_score = wer(reference_chars, hypothesis_chars)

    return wer_score, cer_score

def plot_error_rates(models, wer_scores, cer_scores, title='WER and CER for Whisper Models', save_path='error_rates.png'):
    """
    WER과 CER을 3x2 그리드의 서브플롯으로 시각화하는 함수

    Args:
        models (list): Whisper 모델 이름 리스트.
        wer_scores (list): 각 모델의 WER 리스트.
        cer_scores (list): 각 모델의 CER 리스트.
        title (str): 전체 그래프 제목.
        save_path (str): 저장할 이미지 파일 경로.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        wer = wer_scores[idx] * 100
        cer = cer_scores[idx] * 100
        labels = ['WER', 'CER']
        errors = [wer, cer]
        colors = ['skyblue', 'salmon']

        bars = ax.bar(labels, errors, color=colors)
        ax.set_ylim(0, max(errors) + 10)
        ax.set_ylabel('Error Rate (%)')
        ax.set_title(f'{model} Model')

        # 막대 위에 퍼센트 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.2f}%', ha='center', va='bottom')

    # 남는 서브플롯 비우기
    for i in range(len(models), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 그래프를 이미지 파일로 저장
    plt.savefig(save_path)
    logging.info(f"Plot saved as {save_path}")

    plt.show()

def save_metrics_to_csv(models, wer_scores, cer_scores, filename='error_rates.csv'):
    """
    WER과 CER을 CSV 파일로 저장하는 함수

    Args:
        models (list): Whisper 모델 이름 리스트.
        wer_scores (list): 각 모델의 WER 리스트.
        cer_scores (list): 각 모델의 CER 리스트.
        filename (str): 저장할 CSV 파일 이름.
    """
    data = {
        'Model': models,
        'WER (%)': [w * 100 for w in wer_scores],
        'CER (%)': [c * 100 for c in cer_scores]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    logging.info(f"Error rates saved to {filename}")

if __name__ == "__main__":
    # .env 파일 경로 명시적으로 지정 (필요 시 경로 수정)
    load_dotenv()

    # 환경 변수에서 파일 경로 가져오기
    reference_file = os.getenv("REFERENCE_FILE_PATH")
    hypothesis_dir = os.getenv("HYPOTHESIS_DIR_PATH")

    # 환경 변수 출력 (디버깅 용도)
    logging.info(f"Reference File: {reference_file}")
    logging.info(f"Hypothesis Directory: {hypothesis_dir}")

    # 필수 환경 변수 확인
    if not reference_file:
        logging.error("REFERENCE_FILE_PATH environment variable is not set.")
        raise EnvironmentError("REFERENCE_FILE_PATH environment variable is not set.")
    if not hypothesis_dir:
        logging.error("HYPOTHESIS_DIR_PATH environment variable is not set.")
        raise EnvironmentError("HYPOTHESIS_DIR_PATH environment variable is not set.")

    # Reference 파일 존재 여부 확인
    if not os.path.isfile(reference_file):
        logging.error(f"Reference 파일을 찾을 수 없습니다: {reference_file}")
        raise FileNotFoundError(f"Reference 파일을 찾을 수 없습니다: {reference_file}")

    # Hypothesis 디렉터리 존재 여부 확인
    if not os.path.isdir(hypothesis_dir):
        logging.error(f"Hypothesis 디렉터리를 찾을 수 없습니다: {hypothesis_dir}")
        raise FileNotFoundError(f"Hypothesis 디렉터리를 찾을 수 없습니다: {hypothesis_dir}")

    # 모델명 리스트
    model_names = ['tiny', 'base', 'small', 'medium', 'large', 'turbo']

    # 각 모델별 Hypothesis 파일 경로 찾기
    hypothesis_files = {}
    for model in model_names:
        pattern = os.path.join(hypothesis_dir, f"UCD_TEST_{model}.txt")
        files_found = glob.glob(pattern)
        if not files_found:
            logging.error(f"Hypothesis 파일을 찾을 수 없습니다 ({model}): {pattern}")
            raise FileNotFoundError(f"Hypothesis 파일을 찾을 수 없습니다 ({model}): {pattern}")
        # Assuming only one file per model
        hypothesis_files[model] = files_found[0]
        logging.info(f"Hypothesis File ({model}): {hypothesis_files[model]}")

    # WER과 CER 계산
    logging.info("WER과 CER 계산 시작...")
    models = []
    wer_scores = []
    cer_scores = []

    for model in model_names:
        path = hypothesis_files[model]
        logging.info(f"Processing model: {model}")
        wer_score, cer_score = compute_metrics(reference_file, path)  # 변수 이름 변경
        logging.info(f"{model.capitalize()} Model - WER: {wer_score * 100:.2f}%, CER: {cer_score * 100:.2f}%")
        models.append(model.capitalize())
        wer_scores.append(wer_score)
        cer_scores.append(cer_score)

    # 에러율 시각화 및 저장
    logging.info("에러율 시각화 중...")
    plot_error_rates(
        models, 
        wer_scores, 
        cer_scores, 
        title='WER and CER for Whisper Models', 
        save_path='WER_CER_Whisper_Models.png'  # 저장할 이미지 파일 이름 지정
    )

    # 에러율 데이터 저장
    save_metrics_to_csv(models, wer_scores, cer_scores, filename='error_rates.csv')
