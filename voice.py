
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import pyttsx3


# =====================================================
# WHISPER MODEL
# =====================================================

model = whisper.load_model("base")


# =====================================================
# RECORD AUDIO
# =====================================================

def record_audio(filename="input.wav", duration=15, samplerate=16000):

    print("🎤 Listening...")

    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    wav.write(
        filename,
        samplerate,
        audio
    )

    print("✅ Recording complete!")


# =====================================================
# SPEECH TO TEXT
# =====================================================

def speech_to_text(filename="input.wav"):

    print("🧠 Converting speech to text...")

    result = model.transcribe(
        filename,
        fp16=False
    )

    text = result["text"].strip()

    print("🗣️ You said:", text)

    return text


# =====================================================
# TEXT TO SPEECH
# =====================================================

def text_to_speech(text):

    print("🔊 Speaking...")

    engine = pyttsx3.init()

    engine.say(text)

    engine.runAndWait()

    engine.stop()

    print("✅ Done!")