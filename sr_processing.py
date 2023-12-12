import speech_recognition as sr
from pathlib import Path
import os

PROJ_ROOT: Path = Path().resolve()
MODEL_PATH: str = os.path.join(PROJ_ROOT, "models")


def whisper_recognition(recognizer: sr.Recognizer, 
                        audio: sr.AudioData, 
                        model: str, 
                        language: str | None = None, 
                        prompt: str | None = None):
    """
    使用 Whisper 识别语音内容

    :param recognizer: speech_recognition.Recognizer
    :param audio: 音频数据
    :param model: Whisper 模型
    :param language: 音频内语言, 默认由 Whisper 判断
    :param prompt: 提示词, 默认为空，需要以简体中文展示时使用
    """
    try:
        results = recognizer.recognize_whisper(audio_data=audio, 
                                      model=model, 
                                      language=language,
                                      load_options={"download_root": MODEL_PATH}, 
                                      initial_prompt=prompt)
        print("Whisper recognition result: " + results)
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Whisper")


def audio_recognition(recognizer: sr.Recognizer, file_path: str, model: str):
    """
    使用音频文件作为语言输入来源

    :param recognizer: speech_recognition.Recognizer
    :param file_path: 文件路径
    :param model: Whisper 模型
    """
    with sr.AudioFile(file_path) as source:
        audio: sr.AudioData = recognizer.record(source)  # 读取整个文件
        
    whisper_recognition(recognizer, audio, model)


def microphone_recognition(recognizer: sr.Recognizer, model: str):
    """
    使用麦克风作为语言输入来源

    :param recognizer: speech_recognition.Recognizer
    :param model: Whisper 模型
    """
    with sr.Microphone() as source:
        print("Say something:")
        audio: sr.AudioData = r.listen(source)
        
        whisper_recognition(recognizer, audio, model)


if __name__ == "__main__":
    r = sr.Recognizer()
    
    # 音频文件输入识别
    # audio_path = os.path.join(PROJ_ROOT, "input", "chinese.flac")
    # audio_recognition(r, audio_path, "small")
    
    # 麦克风输入识别
    microphone_recognition(r, "small")
