# Whisper 语音识别模型 Demo

- Whisper 模型[下载](https://github.com/openai/whisper/blob/main/whisper/__init__.py)
- Whisper 支持语言[列表](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)
- Faster Whisper 模型[列表](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/utils.py)、[下载](https://huggingface.co/Systran)

## 安装

1. 配置 CUDA，安装 FFmpeg。
2. 从 Pytoch [官网](https://download.pytorch.org/whl/)下载对应版本的 torch、torchvision、torchaudio 等，或者 `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
3. 创建虚拟环境，安装依赖：`pip install -r requirements.txt`
