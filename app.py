import os
import time
import tempfile
import queue
from collections import deque
import sqlite3
import uuid
from datetime import datetime
import numpy as np
import streamlit as st

try:
    import whisper
except Exception as exc:
    whisper = None
    WHISPER_IMPORT_ERROR = str(exc)
else:
    WHISPER_IMPORT_ERROR = None

try:
    import pyttsx3
except Exception as exc:
    pyttsx3 = None
    PYTTSX3_IMPORT_ERROR = str(exc)
else:
    PYTTSX3_IMPORT_ERROR = None

from scipy.io import wavfile
from scipy.signal import resample_poly

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
except Exception as exc:
    webrtc_streamer = None
    WebRtcMode = None
    WEBRTC_IMPORT_ERROR = str(exc)
else:
    WEBRTC_IMPORT_ERROR = None

from llm import ask_llm, execute_tool, final_answer


# =====================================================
# PERSISTENT CHAT HISTORY (SQLite)
# =====================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")


def get_db_connection():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_chat_database():
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)


def create_conversation():
    conversation_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    with get_db_connection() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, "New Chat", now, now),
        )
    return conversation_id


def list_conversations():
    with get_db_connection() as connection:
        rows = connection.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return rows


def load_conversation_messages(conversation_id):
    with get_db_connection() as connection:
        rows = connection.execute(
            "SELECT role, content FROM chat_messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def persist_turn(conversation_id, user_text, assistant_text):
    now = datetime.now().isoformat(timespec="seconds")
    with get_db_connection() as connection:
        connection.executemany(
            "INSERT INTO chat_messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            [
                (conversation_id, "user", str(user_text), now),
                (conversation_id, "assistant", str(assistant_text), now),
            ],
        )
        row = connection.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row and row[0] == "New Chat":
            title = str(user_text).strip().replace("\n", " ")[:42] or "New Chat"
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conversation_id),
            )
        else:
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )


def delete_conversation(conversation_id):
    with get_db_connection() as connection:
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def clear_conversation_messages(conversation_id):
    now = datetime.now().isoformat(timespec="seconds")
    with get_db_connection() as connection:
        connection.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
        connection.execute(
            "UPDATE conversations SET title = 'New Chat', updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )


init_chat_database()


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Nova AI | Voice Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# PROFESSIONAL UI THEME
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --nova-bg: #0b1020; --nova-panel: #121a2b; --nova-line: rgba(148,163,184,.18); --nova-text: #e8eefb; --nova-muted: #9aa8c2; --nova-accent: #8b5cf6; --nova-blue: #38bdf8; }
    .stApp { background: radial-gradient(ellipse at 15% 0%, rgba(91,65,180,.18), transparent 35%), linear-gradient(180deg,#0b1020 0%,#0b1220 100%); color: var(--nova-text); font-family: 'DM Sans', sans-serif; }
    [data-testid="stHeader"] { background: rgba(11,16,32,.65); }
    [data-testid="stSidebar"] { background: #0d1424; border-right: 1px solid var(--nova-line); }
    [data-testid="stSidebar"] > div { padding-top: 1.4rem; }
    h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
    h1 { color: #f4f7ff !important; }
    p, label, .stMarkdown { color: #dce5f5; }
    [data-testid="stChatMessage"] { background: rgba(18,26,43,.72); border: 1px solid var(--nova-line); border-radius: 18px; padding: 1rem 1.1rem; margin-bottom: .8rem; }
    [data-testid="stChatInput"] { border: 1px solid rgba(139,92,246,.48); border-radius: 18px; background: rgba(18,26,43,.95); box-shadow: 0 8px 30px rgba(0,0,0,.18); }
    [data-testid="stChatInput"] textarea { color: #f4f7ff !important; }
    .stButton > button { border-radius: 11px; border: 1px solid var(--nova-line); min-height: 2.6rem; font-weight: 600; transition: all .18s ease; }
    .stButton > button:hover { border-color: #8b5cf6; color: white; transform: translateY(-1px); }
    div[data-testid="stAlert"] { border-radius: 14px; }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { border-color: var(--nova-line) !important; border-radius: 18px !important; }
    .nova-brand { display:flex; align-items:center; gap:12px; padding: 5px 0 20px; }
    .nova-logo { display:flex; align-items:center; justify-content:center; width:48px; height:48px; border-radius:15px; background:linear-gradient(135deg,#8b5cf6,#38bdf8); box-shadow:0 8px 25px rgba(139,92,246,.25); font-size:25px; }
    .nova-brand-title { font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; color:#f4f7ff; line-height:1.1; }
    .nova-brand-sub { font-size:11px; color:#9aa8c2; margin-top:5px; letter-spacing:.12em; text-transform:uppercase; }
    .nova-hero { padding: 27px 30px; border:1px solid rgba(148,163,184,.2); border-radius:24px; background:linear-gradient(120deg,rgba(139,92,246,.17),rgba(56,189,248,.07) 58%,rgba(18,26,43,.7)); margin: 3px 0 22px; }
    .nova-eyebrow { color:#a5b4fc; font-size:11px; font-weight:700; letter-spacing:.17em; text-transform:uppercase; margin-bottom:10px; }
    .nova-hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(28px,3vw,42px); font-weight:700; line-height:1.12; color:#f8faff; margin-bottom:10px; }
    .nova-hero-copy { color:#aab8d0; font-size:15px; max-width:720px; line-height:1.65; }
    .nova-chip { display:inline-block; border:1px solid rgba(148,163,184,.2); border-radius:999px; padding:6px 10px; margin:14px 6px 0 0; color:#cbd5e1; font-size:12px; background:rgba(15,23,42,.5); }
    .nova-section { color:#f1f5f9; font-family:'Space Grotesk',sans-serif; font-size:18px; font-weight:600; margin: 20px 0 10px; }
    .nova-feature { border:1px solid var(--nova-line); background:rgba(18,26,43,.66); border-radius:16px; padding:15px 16px; min-height:105px; }
    .nova-feature-icon { font-size:20px; margin-bottom:8px; }
    .nova-feature-title { color:#eef2ff; font-weight:700; font-size:14px; margin-bottom:4px; }
    .nova-feature-copy { color:#9aa8c2; font-size:12px; line-height:1.45; }
    .nova-muted { color:#9aa8c2; font-size:13px; }
    footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="nova-brand">
  <div class="nova-logo">🤖</div>
  <div><div class="nova-brand-title">Nova AI</div><div class="nova-brand-sub">Voice assistant workspace</div></div>
</div>
<div class="nova-hero">
  <div class="nova-eyebrow">Your personal AI workspace</div>
  <div class="nova-hero-title">Talk naturally. Get things done.</div>
  <div class="nova-hero-copy">A local AI assistant powered by Gemma 3 and Whisper. Chat by typing or use your microphone for hands-free conversations, quick tools, and contextual follow-ups.</div>
  <span class="nova-chip">✦ Gemma 3</span><span class="nova-chip">◉ Whisper speech-to-text</span><span class="nova-chip">↗ Tool calling</span><span class="nova-chip">▤ Saved chat history</span>
</div>
""", unsafe_allow_html=True)


# =====================================================
# SETTINGS
# =====================================================

SAMPLE_RATE = 16000

SILENCE_SECONDS = 1.5
MIN_SPEECH_SECONDS = 0.5
MAX_SPEECH_SECONDS = 15

VOICE_THRESHOLD = 0.003
PRE_ROLL_SECONDS = 0.4



# =====================================================
# LOAD WHISPER
# =====================================================

@st.cache_resource
def load_whisper():
    if whisper is None:
        raise RuntimeError(
            "Whisper is not installed or could not be imported. "
            f"Original error: {WHISPER_IMPORT_ERROR}"
        )

    return whisper.load_model("small")


# =====================================================
# TEXT TO SPEECH
# =====================================================
def speak_text(text, processor):
    """Speak the answer and temporarily ignore microphone frames."""
    if pyttsx3 is None:
        return

    processor.speaking = True
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(str(text))
        engine.runAndWait()
        engine.stop()
    finally:
        processor.speaking = False


# =====================================================
# LIVE MICROPHONE PROCESSOR
# =====================================================
class LiveAudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.speaking = False
        self.current_audio = []
        self.pre_roll_audio = deque()
        self.speech_started = None
        self.last_voice_time = None
        self.current_sample_rate = SAMPLE_RATE

    def reset_utterance(self):
        self.current_audio = []
        self.pre_roll_audio.clear()
        self.speech_started = None
        self.last_voice_time = None

    def recv(self, frame):
        """Called repeatedly as microphone audio arrives."""
        if self.speaking:
            self.reset_utterance()
            return frame

        raw_audio = frame.to_ndarray()
        if raw_audio.ndim == 2:
            audio = raw_audio.astype(np.float32).mean(axis=0)
        else:
            audio = raw_audio.astype(np.float32).reshape(-1)

        # Convert integer PCM audio to approximately -1..1.
        if np.issubdtype(raw_audio.dtype, np.integer):
            max_value = max(abs(np.iinfo(raw_audio.dtype).min), np.iinfo(raw_audio.dtype).max)
            audio = audio / max_value

        sample_rate = frame.sample_rate or SAMPLE_RATE
        now = time.monotonic()
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        has_voice = rms > VOICE_THRESHOLD

        self.pre_roll_audio.append((now, audio.copy()))
        while self.pre_roll_audio and now - self.pre_roll_audio[0][0] > PRE_ROLL_SECONDS:
            self.pre_roll_audio.popleft()

        if has_voice:
            if self.speech_started is None:
                self.speech_started = now
                # Include the quiet lead-in so the first syllable is not clipped.
                self.current_audio = [chunk for _, chunk in self.pre_roll_audio]
            else:
                self.current_audio.append(audio.copy())
            self.last_voice_time = now
            self.current_sample_rate = sample_rate
        elif self.speech_started is not None:
            # Keep a little trailing silence so the final word is preserved.
            self.current_audio.append(audio.copy())

        if self.speech_started is not None:
            duration = now - self.speech_started
            silence_duration = (
                now - self.last_voice_time if self.last_voice_time is not None else 0
            )
            finished = (
                silence_duration >= SILENCE_SECONDS
                and duration >= MIN_SPEECH_SECONDS
            )
            too_long = duration >= MAX_SPEECH_SECONDS

            if finished or too_long:
                if self.current_audio:
                    utterance = np.concatenate(self.current_audio)
                    self.audio_queue.put((utterance, self.current_sample_rate))
                self.reset_utterance()

        return frame


# =====================================================
# TRANSCRIBE AUDIO
# =====================================================
def transcribe_audio(model, audio, source_rate, language=None):
    """Convert collected browser audio into text using Whisper."""
    if source_rate != SAMPLE_RATE:
        from math import gcd
        common = gcd(int(source_rate), SAMPLE_RATE)
        audio = resample_poly(
            audio,
            SAMPLE_RATE // common,
            int(source_rate) // common,
        )

    audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            temp_path = temp.name
        wavfile.write(temp_path, SAMPLE_RATE, audio_int16)
        initial_prompt = {
            "hi": "यह बातचीत हिंदी और Hinglish में है।",
            "en": "The speaker is speaking English.",
        }.get(language)
        result = model.transcribe(
            temp_path,
            fp16=False,
            language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            temperature=0.0,
            beam_size=5,
            verbose=False,
        )
        return result.get("text", "").strip()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================
# CONVERSATION MEMORY + LLM TOOL CALLING
# =====================================================
def process_user_message(message):
    """Answer using recent conversation history and existing tool calling."""
    # Memory contains only completed previous turns; current message is not added yet.
    history = st.session_state.memory[-10:]

    if history:
        history_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in history
        )
        prompt = f"""You are a helpful AI voice assistant.
Use the previous conversation to understand follow-up questions and pronouns such as 'it', 'that', or 'same'.
Answer in the language the user is using (Hindi/Hinglish or English).
Do not invent facts that are not supported by the conversation.
If the current request needs a tool, follow the tool-calling format expected by the application.

Previous conversation:
{history_text}

Current user message:
{message}
"""
    else:
        prompt = message

    response = ask_llm(prompt)
    if not response:
        return "Sorry, I could not generate an answer."

    response = str(response).strip()
    if response.startswith("TOOL_CALL"):
        result = execute_tool(response)
        answer = final_answer(message, result)
        return str(answer or "The tool did not return an answer.")

    return response


def save_turn(user_text, assistant_text):
    """Save a completed turn to the visible chat, short-term memory, and SQLite."""
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.session_state.memory.append({"role": "user", "content": user_text})
    st.session_state.memory.append({"role": "assistant", "content": assistant_text})
    st.session_state.memory = st.session_state.memory[-20:]
    persist_turn(st.session_state.current_conversation_id, user_text, assistant_text)



# =====================================================
# SESSION STATE
# =====================================================
if "current_conversation_id" not in st.session_state:
    existing_conversations = list_conversations()
    if existing_conversations:
        st.session_state.current_conversation_id = existing_conversations[0][0]
    else:
        st.session_state.current_conversation_id = create_conversation()

if "messages" not in st.session_state:
    st.session_state.messages = load_conversation_messages(
        st.session_state.current_conversation_id
    )
if "memory" not in st.session_state:
    st.session_state.memory = st.session_state.messages[-20:]
if "audio_processor" not in st.session_state:
    st.session_state.audio_processor = LiveAudioProcessor()


# =====================================================
# SIDEBAR + SAVED CONVERSATIONS
# =====================================================
with st.sidebar:
    st.markdown("""
    <div class="nova-brand" style="padding-bottom:10px">
      <div class="nova-logo" style="width:40px;height:40px;font-size:20px;border-radius:12px">✦</div>
      <div><div class="nova-brand-title" style="font-size:18px">Nova AI</div><div class="nova-brand-sub">Control center</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Workspace")
    if st.button("＋  Start a new chat", use_container_width=True, type="primary"):
        st.session_state.current_conversation_id = create_conversation()
        st.session_state.messages = []
        st.session_state.memory = []
        st.rerun()

    conversations = list_conversations()
    if conversations:
        st.markdown("### Your conversations")
        conversation_ids = [row[0] for row in conversations]
        titles = {row[0]: row[1] for row in conversations}
        current_id = st.session_state.current_conversation_id
        if current_id not in conversation_ids:
            current_id = conversation_ids[0]
            st.session_state.current_conversation_id = current_id
            st.session_state.messages = load_conversation_messages(current_id)
            st.session_state.memory = st.session_state.messages[-20:]

        selected_id = st.selectbox(
            "Saved chats",
            options=conversation_ids,
            index=conversation_ids.index(current_id),
            format_func=lambda item: titles.get(item, "Chat")[:42],
            help="Choose a previous conversation to continue.",
            label_visibility="collapsed",
        )
        if selected_id != st.session_state.current_conversation_id:
            st.session_state.current_conversation_id = selected_id
            st.session_state.messages = load_conversation_messages(selected_id)
            st.session_state.memory = st.session_state.messages[-20:]
            st.rerun()

        left, right = st.columns(2)
        with left:
            if st.button("Clear chat", use_container_width=True):
                clear_conversation_messages(st.session_state.current_conversation_id)
                st.session_state.messages = []
                st.session_state.memory = []
                st.rerun()
        with right:
            if st.button("Delete chat", use_container_width=True):
                delete_conversation(st.session_state.current_conversation_id)
                st.session_state.current_conversation_id = create_conversation()
                st.session_state.messages = []
                st.session_state.memory = []
                st.rerun()

    st.divider()
    st.markdown("### System status")
    st.markdown("<div class='nova-muted'>● Local LLM · Ollama Gemma 3</div><div class='nova-muted'>● Speech recognition · Whisper</div><div class='nova-muted'>● Voice output · pyttsx3</div><div class='nova-muted'>● History · SQLite on this computer</div>", unsafe_allow_html=True)
    st.caption("Tip: speak clearly, then pause briefly after each command.")


# =====================================================
# CHAT HISTORY + TEXT INPUT
# =====================================================
if not st.session_state.messages:
    st.markdown('<div class="nova-section">Start with a prompt</div>', unsafe_allow_html=True)
    prompt_cols = st.columns(3)
    starter_prompts = [
        ("🧮", "Quick calculation", "Calculate 245 * 18"),
        ("🌦️", "Check the weather", "What is the weather in Sonipat?"),
        ("🔎", "Search the web", "Search the web for the latest AI news"),
    ]
    for col, (emoji, title, prompt) in zip(prompt_cols, starter_prompts):
        with col:
            st.markdown(f'<div class="nova-feature"><div class="nova-feature-icon">{emoji}</div><div class="nova-feature-title">{title}</div><div class="nova-feature-copy">Click below to send a sample request.</div></div>', unsafe_allow_html=True)
            if st.button(f"Try: {title}", key=f"starter_{title}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

    st.markdown('<div class="nova-muted" style="margin-top:12px">Or type your own question below. Use the microphone panel for hands-free voice chat.</div>', unsafe_allow_html=True)

chat_area = st.container()
with chat_area:
    for chat_message in st.session_state.messages:
        with st.chat_message(chat_message["role"], avatar="🧑" if chat_message["role"] == "user" else "🤖"):
            st.markdown(chat_message["content"])

typed_message = st.chat_input("Message Nova AI…")
if not typed_message and st.session_state.get("pending_prompt"):
    typed_message = st.session_state.pop("pending_prompt")
if typed_message:
    try:
        with st.spinner("Nova is thinking…"):
            answer = process_user_message(typed_message)
        save_turn(typed_message, answer)
        with chat_area:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(typed_message)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)
    except Exception as exc:
        st.error(f"Could not process your message: {exc}")
        st.exception(exc)


# =====================================================
# LIVE MICROPHONE
# =====================================================
st.markdown('<div class="nova-section">🎙️ Voice studio</div>', unsafe_allow_html=True)
st.markdown('<div class="nova-muted">Hands-free mode · Speak, pause, and Nova will respond aloud.</div>', unsafe_allow_html=True)

if webrtc_streamer is None:
    st.error(
        "streamlit-webrtc is not available. Install project requirements and restart the app. "
        f"Original error: {WEBRTC_IMPORT_ERROR}"
    )
    st.stop()

with st.container(border=True):
    st.markdown("**Live microphone**")
    st.caption("अपनी बोली हुई भाषा चुनें। आवाज़ पहचानने के बाद Nova सुना हुआ text दिखाएगा।")
    recognition_mode = st.selectbox(
        "Voice recognition language",
        ("Auto-detect", "Hindi / Hinglish", "English"),
        help="अगर Auto-detect आपकी बात गलत समझे, तो Hindi / Hinglish या English चुनकर फिर बोलें।",
    )

recognition_language = {
    "Auto-detect": None,
    "Hindi / Hinglish": "hi",
    "English": "en",
}[recognition_mode]

processor = st.session_state.audio_processor
ctx = webrtc_streamer(
    key="jarvis-live-microphone",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=processor.recv,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

status = st.empty()

if ctx.state.playing:
    status.info("⏳ Whisper speech-recognition model load हो रहा है...")
    model = load_whisper()
    status.success("🎧 Listening for your voice...")

    while ctx.state.playing:
        try:
            audio, source_rate = processor.audio_queue.get(timeout=0.25)
        except queue.Empty:
            continue

        status.info("🧠 Whisper आपकी आवाज़ को text में बदल रहा है...")
        try:
            transcript = transcribe_audio(
                model,
                audio,
                source_rate,
                language=recognition_language,
            )
            if not transcript:
                status.warning("आवाज़ समझ नहीं आई। कृपया थोड़ा पास और साफ़ बोलकर फिर कोशिश करें।")
                continue

            status.info(
                f"✅ सुना: “{transcript}”\n"
                "🤖 अब Nova जवाब बना रहा है। Local model को कुछ समय लग सकता है..."
            )

            stop_phrases = {
                "stop assistant",
                "goodbye assistant",
                "exit assistant",
            }

            if transcript.lower().strip(" .,!?") in stop_phrases:
                answer = "Goodbye! Use the microphone Stop control to end the session."
                save_turn(transcript, answer)
                with chat_area:
                    with st.chat_message("user"):
                        st.markdown(transcript)
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                speak_text(answer, processor)
                processor.reset_utterance()
                status.warning("Use the microphone Stop control to end listening.")
                break

            status.info("🤖 Gemma जवाब बना रहा है...")
            answer = process_user_message(transcript)
            save_turn(transcript, answer)

            with chat_area:
                with st.chat_message("user"):
                    st.markdown(transcript)
                with st.chat_message("assistant"):
                    st.markdown(answer)

            status.info("🔊 Speaking the answer...")
            speak_text(answer, processor)
            processor.reset_utterance()
            status.success("🎧 Listening for your next command...")

        except Exception as exc:
            status.error(f"Something went wrong: {exc}")
            st.exception(exc)

    status.info("Microphone stopped.")
else:
    status.info("Click START in the microphone control above to begin.")
