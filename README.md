# 🤖 AI Voice Assistant — Real-Time Voice Interaction

An intelligent, voice-enabled AI assistant built with **Python, Streamlit, WebRTC, OpenAI Whisper, and Ollama (Gemma 3)**. The assistant is designed to understand spoken commands, generate AI-powered responses, and perform useful tasks through tool calling.

The project aims to provide a more natural, hands-free conversational experience with support for Hindi and English voice input.

## ✨ Features

- 🎙️ **Voice Input:** Capture microphone audio using WebRTC.
- 🗣️ **Speech-to-Text:** Convert spoken language into text using OpenAI Whisper.
- 🌐 **Hindi & English Support:** Designed for multilingual voice interaction.
- 🧠 **Local LLM Integration:** Generate responses using Gemma 3 through Ollama.
- 🔊 **Text-to-Speech:** Convert AI responses into spoken audio using pyttsx3.
- ⚡ **Continuous Voice Interaction:** Support for ongoing voice conversations with automatic response handling.
- 🛠️ **Tool Calling:** Extend the assistant with tools such as a calculator, date and time, web search, and weather.
- 💬 **Chat History:** Maintain conversation messages within the Streamlit session.
- 🎨 **Interactive UI:** A Streamlit-based interface for interacting with the assistant.

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| WebRTC | Real-time microphone audio streaming |
| OpenAI Whisper | Speech recognition |
| Ollama | Local LLM inference |
| Gemma 3 | AI response generation |
| pyttsx3 | Text-to-speech |
| NumPy & SciPy | Audio processing |
| Requests / DuckDuckGo Search | Web-related tool integrations |

## 🏗️ Architecture

```text
User Voice
    ↓
Microphone (WebRTC)
    ↓
Audio Processing
    ↓
Whisper Speech Recognition
    ↓
Recognized Text
    ↓
Ollama + Gemma 3
    ↓
Tool Calling (if required)
    ↓
AI Response
    ↓
Text-to-Speech (pyttsx3)
    ↓
Spoken Response
```

## 📁 Project Structure

```text
AI-Voice-Assistant/
│
├── app.py              # Streamlit application and UI
├── voice.py            # Audio processing, STT and TTS
├── llm.py              # LLM integration and tool calling
├── requirements.txt    # Python dependencies
├── .gitignore          # Files excluded from Git
└── README.md           # Project documentation
```

*Your actual filenames may differ depending on your implementation.*

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Voice-Assistant
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and download the model

Install Ollama from [ollama.com](https://ollama.com/), then run:

```bash
ollama pull gemma3:4b
```

Make sure Ollama is running locally before starting the application.

### 5. Install FFmpeg

Whisper may require FFmpeg for audio decoding. Install FFmpeg and ensure its executable is available in your system PATH.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

Allow microphone access in your browser when prompted.

## 🚀 How It Works

1. The user speaks through the microphone.
2. WebRTC captures audio for processing.
3. Whisper converts the audio into text.
4. Gemma 3 processes the recognized query and generates a response.
5. The application can invoke available tools when required.
6. The generated response is converted into speech.
7. The assistant continues the interaction according to the configured audio and silence-detection logic.

## 🔮 Future Enhancements

- Persistent conversation memory across sessions.
- Improved voice activity detection and noise reduction.
- Better multilingual transcription and language selection.
- Wake-word activation, such as “Hey Jarvis.”
- More tools and API integrations.
- Customizable assistant voice and speaking speed.
- Conversation export and user preferences.
- Optimized latency for smoother real-time interaction.




⭐ If you find this project interesting, consider giving the repository a star!
