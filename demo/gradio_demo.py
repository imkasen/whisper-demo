"""
UI Demo
"""

import os
import tempfile

import gradio as gr
import whisper
from whisper.utils import get_writer


def audio_extraction(video_path: str) -> str:
    """
    利用 FFmpeg 提取视频中的音频文件

    :param video_path: 视频文件路径
    :return: 提取的音频文件路径
    """
    video_name, _ = os.path.splitext(video_path)
    _, video_name = os.path.split(video_name)
    audio_path: str = os.path.join(tempfile.gettempdir(), f"{video_name}.aac")
    os.system(f"ffmpeg -hide_banner -v error -i {video_path} -vn -c:a copy {audio_path} -y")
    return audio_path


def transcribe(
    audio_path: str,
    model_size: str,
    output_format: str,
    prompt: str | None,
    language: str | None,
) -> tuple[str | list, list[str]] | tuple[str | list, str]:
    """
    使用 whisper 识别音频文件内语音，并生成字幕文件。

    :param audio_path: 音频文件路径
    :param model_size: 模型名称
    :param output_format: 输出文件格式
    :param prompt: 提示词
    :param language: 音频文件所使用的语言，默认为空，即由 whisper 负责识别
    """
    if not prompt:
        prompt = None
    if not language:
        language = None

    file_name, _ = os.path.splitext(audio_path)
    _, file_name = os.path.split(file_name)

    model: whisper.Whisper = whisper.load_model(model_size)
    result: dict[str, str | list] = model.transcribe(audio_path, language=language, initial_prompt=prompt)

    writer = get_writer(output_format, tempfile.gettempdir())
    writer(result, file_name)

    if output_format == "all":
        tmp_lst: list[str] = []
        for file_format in ["txt", "vtt", "srt", "tsv", "json"]:
            tmp_lst.append(os.path.join(tempfile.gettempdir(), f"{file_name}.{file_format}"))
        return result["text"], tmp_lst
    return result["text"], os.path.join(tempfile.gettempdir(), f"{file_name}.{output_format}")


# Gradio UI
with gr.Blocks(title="Whisper Gradio Demo") as demo:
    gr.HTML(value="<h1 align='center'>Whisper Gradio Demo</h1>")

    with gr.Tab(label="Whisper"):
        with gr.Row():
            with gr.Column(variant="panel"):
                whisper_model_size = gr.Dropdown(
                    label="模型",
                    choices=["tiny", "base", "small", "medium", "large"],
                    value="large",
                    interactive=True,
                )
                whisper_output_format = gr.Dropdown(
                    label="输出格式",
                    choices=["txt", "vtt", "srt", "tsv", "json", "all"],
                    value="srt",
                    interactive=True,
                )
                whisper_prompt = gr.Textbox(
                    label="提示词",
                    value="如果使用了中文，请使用简体中文来表示文本内容。",
                    interactive=True,
                )
                whisper_language = gr.Textbox(
                    label="语言",
                    info="音频文件所使用的语言，默认为空，即由 whisper 负责识别。",
                    interactive=True,
                )
            with gr.Column(variant="panel"):
                whisper_upload_video = gr.Video(label="步骤 1", include_audio=True, interactive=True)
                whisper_video_submit = gr.Button(value="转换为音频")
            with gr.Column(variant="panel"):
                whisper_upload_audio = gr.Audio(label="步骤 2", interactive=True, type="filepath")
                whisper_audio_submit = gr.Button(value="转录", variant="primary")
        with gr.Row():
            whisper_result_text = gr.Textbox(interactive=False, container=False)
            whisper_result_file = gr.File(interactive=False)

        whisper_video_submit.click(  # pylint: disable=E1101
            fn=audio_extraction,
            inputs=whisper_upload_video,
            outputs=whisper_upload_audio,
        )
        whisper_audio_submit.click(  # pylint: disable=E1101
            fn=transcribe,
            inputs=[
                whisper_upload_audio,
                whisper_model_size,
                whisper_output_format,
                whisper_prompt,
                whisper_language,
            ],
            outputs=[whisper_result_text, whisper_result_file],
        )


if __name__ == "__main__":
    demo.queue().launch(
        inbrowser=True,
        share=False,
        show_api=False,
    )
