from faster_whisper import WhisperModel
import os
import time
from pathlib import Path

PROJ_ROOT: Path = Path().resolve()
MODEL_PATH: str = os.path.join(PROJ_ROOT, "models", "faster_whisper")
PROMPT = "如果使用了中文，请使用简体中文来表示文本内容"


def extraction(video_path: str, output_dir_path: str):
    """
    利用 FFmpeg 提取视频中的音频文件

    :param video_path: 视频文件路径
    :return: 提取的音频文件路径
    """
    video_name, _ = os.path.splitext(video_path)
    _, video_name = os.path.split(video_name)
    audio_path: str = os.path.join(output_dir_path, f"{video_name}.aac")
    os.system(f"ffmpeg -hide_banner -v error -i {video_path} -vn -c:a copy {audio_path} -y")
    return audio_path


def transcribe(audio_path: str, model_size: str, language: str | None = None, prompt: str | None = None):
    """
    使用 faster whisper 识别音频文件内语音。

    :param audio_path: 音频文件路径
    :param model_size: 模型名称
    :param language: 音频文件所使用的语言，默认为空，即由 whisper 负责识别
    :param prompt: 提示词，默认为空
    """
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16", download_root=MODEL_PATH, local_files_only=True)

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16", download_root=MODEL_PATH, local_files_only=True)
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # ================
    # BASIC USAGE
    segments, info = model.transcribe(audio_path, beam_size=5, language=language, initial_prompt=prompt)
    
    if language is None:
        print(f"Detected language '{info.language}' with probability {info.language_probability}")

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    
    # ================
    # WORD-LEVEL TIMESTAMPS
    # segments, _ = model.transcribe(audio_path, word_timestamps=True, language=language, initial_prompt=prompt)

    # for segment in segments:
    #     for word in segment.words:
    #         print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
    
    # ================
    # VAD FILTER
    # segments, _ = model.transcribe(
    #     audio_path,
    #     language=language,
    #     initial_prompt=prompt,
    #     vad_filter=True,
    #     vad_parameters=dict(min_silence_duration_ms=500)
    # )
    
    # for segment in segments:
    #     print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")


if __name__ == "__main__":
    t1: float = time.perf_counter()
    
    VIDEO_PATH: str = os.path.join(PROJ_ROOT, "input", "zh.mp4")
    
    audio_path: str = extraction(VIDEO_PATH, PROJ_ROOT)
    transcribe(audio_path, "large", None, PROMPT)
    
    os.remove(audio_path)
    
    t2: float = time.perf_counter()
    print(f"Total time: {(t2 - t1):.2f}s")
