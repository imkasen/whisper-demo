"""
使用 Whisper、SpeechRecognition 对音频文件、麦克风输入进行语言识别
"""

import speech_recognition as sr
from pathlib import Path
from threading import Thread
from queue import Queue
from faster_whisper import WhisperModel
import os
import io
import soundfile as sf
import numpy as np

PROJ_ROOT: Path = Path().resolve()
MODEL_PATH: str = os.path.join(PROJ_ROOT, "models")
PROMPT = "如果使用了中文，请使用简体中文来表示文本内容"


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
        results: str = recognizer.recognize_whisper(audio_data=audio, 
                                      model=model, 
                                      language=language,
                                      load_options={"download_root": MODEL_PATH}, 
                                      initial_prompt=prompt)
        if (results := results.strip()) != "":
            print(results)
    except sr.UnknownValueError:
        print("Whisper could not understand audio!")
    except sr.RequestError as e:
        print("Could not request results from Whisper!")


def faster_whisper_recognition(audio: sr.AudioData, 
                               model_size: str, 
                               language: str | None = None, 
                               prompt: str | None = None):
    try:
        wav_bytes: bytes = audio.get_wav_data(convert_rate=16000)
        wav_stream = io.BytesIO(wav_bytes)
        audio_array, _ = sf.read(wav_stream)
        audio_array = audio_array.astype(np.float32)
        
        model = WhisperModel(model_size, 
                             device="cuda", 
                             compute_type="float16", 
                             download_root=os.path.join(MODEL_PATH, "faster_whisper"), 
                             local_files_only=True)
        
        segments, _ = model.transcribe(
            audio_array, 
            language=language,
            initial_prompt=prompt
        )
        
        for segment in segments:
            print(segment.text)
        
    except sr.UnknownValueError:
        print("Whisper could not understand audio!")
    except sr.RequestError as e:
        print("Could not request results from Whisper!")


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
    使用麦克风作为语言输入来源，单次识别

    :param recognizer: speech_recognition.Recognizer
    :param model: Whisper 模型
    """
    with sr.Microphone() as source:
        print("Say something:")
        audio: sr.AudioData = r.listen(source)
        
        whisper_recognition(recognizer, audio, model, prompt=PROMPT)
        
        
def microphone_thread_worker(recognizer: sr.Recognizer, que: Queue[sr.AudioData], model: str):
    """
    后台运行线程

    :param recognizer: speech_recognition.Recognizer
    :param que: 音频任务队列
    :param model: Whisper 模型
    """
    while True:
        audio: sr.AudioData = que.get()  # 从主线程中获取下一个音频处理任务
        if audio is None:
            break  # 主线程结束时停止处理
        
        # 收到音频数据，现在我们可以识别了
        # whisper_recognition(recognizer, audio, model, prompt=PROMPT)
        faster_whisper_recognition(audio, model, prompt=PROMPT)
        
        que.task_done()  # 在队列中将音频处理作业标记为已完成
        
        
def microphone_background_recognition(recognizer: sr.Recognizer, model: str):
    """
    使用麦克风作为语言输入来源，后台持续监听

    :param recognizer: speech_recognition.Recognizer
    :param model: Whisper 模型
    """
    audio_queue = Queue()
    recognize_thread = Thread(target=microphone_thread_worker, args=(recognizer, audio_queue, model,), daemon=True)
    recognize_thread.start()
    with sr.Microphone(sample_rate=16000) as source:
        recognizer.adjust_for_ambient_noise(source)  # 聆听 1 秒钟来校准环境噪音水平的能量阈值
        print("Say something:")
        try:
            while True:  # 反复聆听，并将生成的音频放入音频处理任务队列
                audio_queue.put(recognizer.listen(source))
        except KeyboardInterrupt:  # Ctrl + C 退出程序
            pass

    audio_queue.join()  # 阻塞直到当前队列内所有音频处理工作完成
    audio_queue.put(None)  # 告诉 recognize_thread 停止
    recognize_thread.join()  # 等待 recognize_thread 真正停止


if __name__ == "__main__":    
    r = sr.Recognizer()
    
    # 音频文件输入识别
    # audio_path = os.path.join(PROJ_ROOT, "input", "chinese.flac")
    # audio_recognition(r, audio_path, "large")
    
    # 麦克风输入识别，只进行单次识别
    # microphone_recognition(r, "large")
    
    # 麦克风输入识别，后台运行线程进行持续监听
    microphone_background_recognition(r, "large")
