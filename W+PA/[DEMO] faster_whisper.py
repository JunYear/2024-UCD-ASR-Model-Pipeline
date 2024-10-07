from faster_whisper import WhisperModel

model_size = "large-v3"

audio_file = r"{file_path}"  # 처리할 음성 파일 경로

# Run on GPU with FP16
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe(audio_file, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print(f"[{segment.start} - {segment.end}]: {segment.text}")