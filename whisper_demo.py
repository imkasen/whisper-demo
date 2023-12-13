"""
使用 Whisper 对视频文件进行音频识别
"""

import whisper
from whisper.utils import get_writer
import os
import time
from pathlib import Path

PROJ_ROOT: Path = Path().resolve()
MODEL_PATH: str = os.path.join(PROJ_ROOT, "models")


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
    print(f"{video_name}.aac is extracted.")
    return audio_path


def transcribe(audio_path: str, model_name: str = "small", language: str | None = None, prompt: str | None = None):
    """
    使用 whisper 识别音频文件内语音，并生成字幕文件。

    :param audio_path: 音频文件路径
    :param language: 音频文件所使用的语言，默认为空，即由 whisper 负责识别
    :param model_name: 模型名称, 默认为 "small"
    :param prompt: 提示词，默认为空
    :return: 返回字典类型，即经过 whisper 进行 transcribe 后的内容
    """
    # model_name: tiny, base, small, medium, large
    model = whisper.load_model(model_name, download_root=MODEL_PATH)
    result = model.transcribe(audio_path, language=language, initial_prompt=prompt)
    print(f"{audio_path} is transcribed.")
    return result


def write_output(result: dict[str, str | list], file_name: str, output_dir_path: str, output_format: str = "srt"):
    """
    生成输入文件。

    :param result: 字典类型，经过 whisper 进行 transcribe 后的内容
    :param file_name: 输出文件名
    :param output_dir_path: 输出文件夹路径
    :param output_format: 输出文件格式, 默认为 "srt"
    """
    file_path: str = f"{file_name}.{output_format}"
    # output_format: txt, vtt, srt, tsv, json, all
    writer = get_writer(output_format, output_dir_path)
    writer(result, file_path)
    print(f"{file_path} is written.")


if __name__ == "__main__":
    t1: float = time.perf_counter()

    VIDEO_PATH: str = os.path.join(PROJ_ROOT, "input", "zh.mp4")
    FILE_NAME, _ = os.path.splitext(VIDEO_PATH)
    _, FILE_NAME = os.path.split(FILE_NAME)
    OUTPUT_DIR_PATH: str = os.path.join(PROJ_ROOT, "output")
    if not (os.path.exists(OUTPUT_DIR_PATH) and os.path.isdir(OUTPUT_DIR_PATH)):
        os.makedirs(OUTPUT_DIR_PATH)
        print(f"Create {OUTPUT_DIR_PATH}")

    audio_path: str = extraction(VIDEO_PATH, OUTPUT_DIR_PATH)
    # result = transcribe(audio_path, "tiny", "en")
    result = transcribe(audio_path, "small", None, "如果出现中文请使用简体")
    write_output(result, FILE_NAME, OUTPUT_DIR_PATH)

    t2: float = time.perf_counter()
    print(f"Total time: {(t2 - t1):.2f}s")
