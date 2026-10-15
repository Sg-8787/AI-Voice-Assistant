# AI Voice Assistant

A local voice assistant with a Streamlit chat UI, Whisper speech recognition, and an Ollama-powered Gemma model. It also includes calculator, current-time, weather, and web-search tools.

## Requirements

- Python
- [Ollama](https://ollama.com/) with the `gemma3:4b` model

## Setup on Windows

```powershell
git clone https://github.com/Sg-8787/AI-Voice-Assistant.git
cd AI-Voice-Assistant
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull gemma3:4b
streamlit run app.py
```

Keep Ollama running while using the assistant. On first launch, Whisper may need to download its speech-recognition model. Allow microphone access in your browser, select the language you are speaking, then pause briefly after each request.

Chat history is stored locally in `chat_history.db`. Local virtual environments, chat history, and audio recordings are excluded from Git.