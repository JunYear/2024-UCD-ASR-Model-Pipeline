import os
import torch
from pyannote.audio import Pipeline
import whisper
import wave
import tempfile


def format_speaker_label(speaker):
    """
    화자 라벨을 'SPEAKER 01' 형식으로 변환을 위한 함수
    예: 'SPEAKER_00' -> 'SPEAKER 00'
    """
    if speaker.lower().startswith("speaker_"):
        speaker_num = speaker.split("_")[-1]
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
    # 음성 파일 존재 여부 확인
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError("Audio files not found: {audio_file_path}")
    
    # 음성 파일 이름과 확장자 분리
    audio_filename = os.path.basename(audio_file_path)
    base_name, _ = os.path.splitext(audio_filename)

    # 출력 파일 경로 준비
    output_file_path = os.path.join(output_dir, f"{base_name}.txt")

    # 화자 다이어리제이션 파이프라인 로드
    print("Loading speaker diarization pipeline...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=access_token
    )

    # GPU 사용 가능 시 파이프라인을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diarization_pipeline.to(device)

    # 화자 다이어리제이션 수행
    print("Performing speaker diarization...")
    diarization = diarization_pipeline(audio_file_path)

    # Whisper 모델 로드
    print("Loading Whisper Model...")
    whisper_model = whisper.load_model("large-v3")

    # 전사 수행
    print("Performing transcription...")
    transcription = whisper_model.transcribe(audio_file_path, word_timestamps=True)

    # 단어 단위 세그먼트 추출
    words = transcription.get("segments", [])

    # 단어를 화자에 매핑
    print("Mapping words to speakers...")
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
    print("Aggregating text by speaker...")
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
    print("Preparing output text...")
    output_lines = []
    for segment in aggregated_segments:
        speaker = format_speaker_label(segment["speaker"])
        text = segment["text"]
        # 형식: SPEAKER 01: {speeching text}
        output_line = f"{speaker}: {text}"
        output_lines.append(output_line)
    
    # 출력 파일에 쓰기
    print(f"Saving results to a {output_file_path}...")
    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")
    
    print("Processing completed.")


if __name__ == "__main__":
    audio_file = "sample.wav"  # 처리할 음성 파일 경로
    output_directory = "output"  # 텍스트 파일을 저장할 디렉터리
    huggingface_token = "YOUR_AUTH_TOKEN"    # Hugging Face 액세스 토큰
    
    # 출력 디렉터리가 없으면 생성
    os.makedirs(output_directory, exist_ok=True)
    
    # 함수 호출
    diarize_and_transcribe(audio_file, output_directory)
