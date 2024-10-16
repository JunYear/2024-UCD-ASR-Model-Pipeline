import whisper
import os

def transcribe_audio(audio_path, prompt_text):
    if not os.path.isfile(audio_path):
        print(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        return

    model = whisper.load_model("base")
    result = model.transcribe(
        audio_path,
        prompt=prompt_text,
        language="ko",
        task="transcribe",
        beam_size=5,
        temperature=0.0,
        best_of=5,
        patience=1.0,
        length_penalty=1.0,
        condition_on_previous_text=True,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=-1.0,
        word_timestamps=True
    )

    print("인식된 텍스트:")
    print(result["text"])

    print("\n단어별 타임스탬프:")
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f} - {segment['end']:.2f}}] {segment['text']}")

# 사용 예시
audio_file = "path/to/your/audio.wav"
prompt = "이 대화는 개발자들이 회의입니다."
transcribe_audio(audio_file, prompt)
