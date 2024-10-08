import os
import torch
from pyannote.audio import Pipeline
import whisper
import logging
from dotenv import load_dotenv
import time

# 로깅 설정
"""
%(asctime)s: 로그가 기록된 시간 (예: 2024-04-27 12:34:56,789).
%(levelname)s: 로그 레벨 이름 (예: INFO, WARNING).
%(message)s: 실제 로그 메시지 내용.
"""
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)


def format_speaker_label(speaker):
    """
    화자 라벨을 'SPEAKER 01' 형식으로 변환을 위한 함수
    예: 'SPEAKER_00' -> 'SPEAKER 00'
    """
    if speaker.lower().startswith("speaker_"):
        speaker_num = speaker.split("_")[-1].zfill(2)   # 두 자리 숫자로 포맷 (ex. 1 -> 01)
        return f"SPEAKER {speaker_num}"
    else:
        return f"SPEAKER {speaker}"
    
def diarize_and_transcribe(audio_file_path, output_dir, access_token):
    """
    지정된 음성 파일에 대한 화자 다이어리제이션과 전사를 수행하고,
    결과를 지정된 디레거리에 텍스트 파일에 저장하는 함수

    Args:
        audio_file_path (str): 입력 음성 파일의 경로 (예: .wav).
        output_dir (str): 텍스트 파일을 저장할 디렉터리 경로.
        access_token (str): Hugging Face 액세스 토큰.
    """

    # 전체 처리 시간 측정 시작
    start_total = time.perf_counter()

    # 음성 파일 존재 여부 확인
    if not os.path.isfile(audio_file_path):
        logging.error(f"Audio file not found: {audio_file_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    # 음성 파일 이름과 확장자 분리
    audio_filename = os.path.basename(audio_file_path)
    base_name, _ = os.path.splitext(audio_filename)

    # 출력 파일 경로 준비
    output_file_path = os.path.join(output_dir, f"{base_name}.txt")

    # 화자 다이어리제이션 파이프라인 로드
    logging.info("Loading speaker diarization pipeline...")
    start_diarization_pipeline = time.perf_counter()
    
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=access_token
        )
    
    except Exception as e:
        logging.error("Failed to load speaker diarization pipeline: {e}")
        raise e
    
    end_diarization_pipeline = time.perf_counter()
    logging.info(f"Finished loading the speaker diarization pipeline.\n---> Time taken: {end_diarization_pipeline - start_diarization_pipeline:.2f}s")

    # GPU 사용 가능 시 파이프라인을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diarization_pipeline.to(device)
    logging.info(f"Use {device} in pipeline")


    # 화자 다이어리제이션 수행
    logging.info("Performing speaker diarization...")
    start_diarization = time.perf_counter()
    
    try:
        diarization = diarization_pipeline(audio_file_path)
    except Exception as e:
        logging.error("Failed to perform speaker diarization: {e}")
        raise e

    end_diarization = time.perf_counter()
    logging.info(f"Finished Speaker Diarization.\n---> Time taken: {end_diarization - start_diarization:.2f}s")


    # Whisper 모델 로드
    logging.info("Loading Whisper Model...")
    start_transcription_load = time.perf_counter()
    
    try:
        whisper_model = whisper.load_model("tiny")
    except Exception as e:
        logging.error("Failed to load whipser: {e}")
        raise e
    
    end_transcription_load = time.perf_counter()
    logging.info(f"Finished loading the \"Whisper-{whisper_model}\"..\n---> Time taken: {end_transcription_load - start_transcription_load:.2f}s")
 

    # 전사 수행
    logging.info("Performing transcription...")
    start_transcription = time.perf_counter()
    
    try:
        transcription = whisper_model.transcribe(audio_file_path, word_timestamps=True)
    except Exception as e:
        logging.error(f"Failed to perform transcription: {e}")
        raise e
    
    end_transcription = time.perf_counter()
    logging.info(f"Finished transcription.\n---> Time taken: {end_transcription - start_diarization:.2f}s")

    # 단어 단위 세그먼트 추출
    words = transcription.get("segments", [])

    # 단어를 화자에 매핑
    logging.info("Mapping words to speakers...")
    speaker_segments = []

    for word in words:
        word_text = word.get("text", "").strip()
        word_start = word.get("start", 0.0)
        word_end = word.get("end", 0.0)

        # 단어의 시작 시간에 해당하는 화자 찾기
        speaker = "Unknown"
        for turn, _, spk in diarization.itertracks(yield_label=True):
            if turn.start <= word_start < turn.end:
                speaker = spk
                break
                
        speaker_segments.append({
            "speaker": speaker,
            "start": word_start,
            "end": word_end,
            "text": word_text
        })
    
    # 연속된 화자 세그먼트로 텍스트 집계
    logging.info("Aggregating text by speaker...")
    aggregated_segments = []
    current_speaker = None
    current_text = ""

    for segment in speaker_segments:
        speaker = segment["speaker"]
        text = segment["text"]

        if speaker != current_speaker:
            if current_speaker is not None and current_text.strip():
                aggregated_segments.append({
                    "speaker": current_speaker,
                    "text": current_text.strip()
                })
            current_speaker = speaker
            current_text = text + " "
        else:
            current_text += text + " "

    # 마지막 화자 세그먼트 추가    
    if current_speaker is not None and current_text.strip():
        aggregated_segments.append({
            "speaker": current_speaker,
            "text": current_text.strip()
        })
    
    # 출력 텍스트 준비
    logging.info("Preparing output text...")
    output_lines = []
    for segment in aggregated_segments:
        speaker = format_speaker_label(segment["speaker"])
        text = segment["text"]
        # 형식: SPEAKER 01: {speeching text}
        output_line = f"{speaker}: {text}"
        output_lines.append(output_line)
    
    # 출력 파일에 쓰기
    logging.info(f"Saving results to a {output_file_path}...")
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")
    except Exception as e:
        logging.error(f"Failed to save a text file: {e}")
        raise e
    
    # 전체 처리 시간 측정 종료
    end_total = time.perf_counter()
    logging.info(f"Processing completed.\n---> Total Time taken: {end_total - start_total:.2f}s")


if __name__ == "__main__":
    # .env 파일 로드
    load_dotenv()

    # .env에서 환경 변수 가져오기
    audio_file = os.getenv("AUDIO_FILE_PATH", "STT+Diarization/sample.wav")  # 처리할 음성 파일 경로
    output_directory = os.getenv("OUTPUT_DIRECTORY", "output")  # 텍스트 파일을 저장할 디렉터리
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN") # Hugging Face 액세스 토큰

    # 필수 환경 변수 확인
    if not audio_file:
        logging.error("The \"AUDIO_FILE_PATH\" environment variable is not set.")
        raise EnvironmentError("The \"AUDIO_FILE_PATH\" environment variable is not set.")
    
    if not output_directory:
        logging.error("The \"OUTPUT_DIRECTORY\" environment variable is not set.")
        raise EnvironmentError("The \"OUTPUT_DIRECTORY\" environment variable is not set.")
    
    if not huggingface_token:
        logging.error("The \"HUGGINGFACE_TOKEN\" environment variable is not set.")
        raise EnvironmentError("The \"HUGGINGFACE_TOKEN\" environment variable is not set.")
    
    """
    # 출력 디렉터리가 없으면 생성
    os.makedirs(output_directory, exist_ok=True)
    """
    # 함수 호출
    diarize_and_transcribe(audio_file, output_directory, huggingface_token)