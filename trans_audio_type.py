import os
from pydub import AudioSegment

# 지원되는 오디오 파일 확장자 목록
audio_extensions = [
    '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma',
    '.alac', '.aiff', '.ape', '.wav', '.amr', '.opus'
]

# 원본 및 대상 디렉터리 경로 설정
source_directory = '/path/to/source_directory'
target_directory = '/path/to/target_directory'

# 대상 디렉터리가 없다면 생성
os.makedirs(target_directory, exist_ok=True)

# 파일이 지원하는 형식인지 확인
is_supported_audio_file = lambda filename:(
    os.path.isfile(os.path.join(source_directory, filename)) and
    os.path.splitext(filename)[1].lower() in audio_extensions
)


def convert_to_wav(filename):
    """
    파일을 WAV로 변환하는 함수
    예: 'sample.m4a' -> 'sample.wav'
    """
    file_path = os.path.join(source_directory, filename)
    try:
        # 오디오 파일 로드
        audio = AudioSegment.from_file(file_path)
        
        # 파일명과 확장자 분리
        base_name = os.path.splitext(filename)[0]
        
        # 대상 파일 경로 설정
        target_file_path = os.path.join(target_directory, base_name + '.wav')

        # WAV 형식으로 내보내기
        audio.export(target_file_path, format='wav')
        print(f"Conversion Success: {filename} -> {base_name}.wav")
    
    except Exception as e:
        print(f"Conversion Failed: {filename} - 오류 {e}")

if __name__ == '__main__':
    # 원본 디렉터리의 모든 파일 가져오기
    filenames = os.listdir(source_directory)

    # 지원되는 오디오 파일 필터링
    supported_files = filter(is_supported_audio_file, filenames)

    # WAV파일로 변환
    list(map(convert_to_wav, supported_files))
        