"""
使用 Whisper 对视频文件进行音频识别
"""

import os
import time
from pathlib import Path

import whisper
from whisper.utils import get_writer

PROJ_ROOT: Path = Path().resolve()
MODEL_PATH: str = os.path.join(PROJ_ROOT, "models")
PROMPT = "如果使用了中文，请使用简体中文来表示文本内容"


def extraction(video_path: str, output_dir_path: str) -> str:
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


def transcribe(
    audio_path: str,
    model_name: str = "small",
    language: str | None = None,
    prompt: str | None = None,
) -> dict[str, str | list]:
    """
    使用 whisper 识别音频文件内语音，并生成字幕文件。

    :param audio_path: 音频文件路径
    :param language: 音频文件所使用的语言，默认为空，即由 whisper 负责识别
    :param model_name: 模型名称, 默认为 "small"
    :param prompt: 提示词，默认为空
    :return: 返回字典类型，即经过 whisper 进行 transcribe 后的内容
    """
    # model_name: tiny, base, small, medium, large
    # model: whisper.Whisper = whisper.load_model(model_name, download_root=MODEL_PATH)

    # Model path on Windows: C:\Users\<UserName>\.cache\whisper
    model: whisper.Whisper = whisper.load_model(model_name)

    result: dict[str, str | list] = model.transcribe(audio_path, language=language, initial_prompt=prompt)
    print(f"{audio_path} is transcribed.")
    return result


def write_output(result: dict[str, str | list], file_name: str, output_dir_path: str, output_format: str = "srt"):
    """
    生成输出文件。

    :param result: 字典类型，经过 whisper 进行 transcribe 后的内容
    :param file_name: 输出文件名
    :param output_dir_path: 输出文件夹路径
    :param output_format: 输出文件格式, 默认为 "srt"
    """
    # output_format: txt, vtt, srt, tsv, json, all
    writer = get_writer(output_format, output_dir_path)
    writer(result, file_name)
    print(f"{file_name} is written.")


if __name__ == "__main__":
    t1: float = time.perf_counter()

    VIDEO_PATH: str = os.path.join(PROJ_ROOT, "input", "zh.mp4")
    FILE_NAME, _ = os.path.splitext(VIDEO_PATH)
    _, FILE_NAME = os.path.split(FILE_NAME)
    OUTPUT_DIR_PATH: str = os.path.join(PROJ_ROOT, "output")
    if not (os.path.exists(OUTPUT_DIR_PATH) and os.path.isdir(OUTPUT_DIR_PATH)):
        os.makedirs(OUTPUT_DIR_PATH)
        print(f"Create {OUTPUT_DIR_PATH}")

    tmp_audio_path: str = extraction(VIDEO_PATH, OUTPUT_DIR_PATH)
    res: dict[str, str | list] = transcribe(tmp_audio_path, "small", None, PROMPT)
    write_output(res, FILE_NAME, OUTPUT_DIR_PATH)

    os.remove(tmp_audio_path)

    t2: float = time.perf_counter()
    print(f"Total time: {(t2 - t1):.2f}s")
