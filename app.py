import streamlit as st
import os
import wave
import re
import random
import sounddevice as sd
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz
from gtts import gTTS

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="NPS Voice-Activated Utility & Payment Hub",
    page_icon="💳",
    layout="centered"
)

# =====================================================================
# MOCK DATABASES (Nepal Payment Solution)
# =====================================================================
USER_CONTACTS = [
    {"id": "ACC101", "name": "Ram Sharma", "bank": "Nabil Bank"},
    {"id": "ACC102", "name": "Ram Kumar Thapa", "bank": "NIC Asia"},
    {"id": "ACC103", "name": "Ramesh Karki", "bank": "Global IME"},
    {"id": "ACC104", "name": "Sita Devi", "bank": "eSewa Wallet"}
]

UTILITY_PROVIDERS = [
    {"service": "electricity", "keywords": ["electricity", "light", "nea", "buli", "bijuli"], "provider": "Nepal Electricity Authority (NEA)"},
    {"service": "water", "keywords": ["water", "khanepani", "pani"], "provider": "Kathmandu Upatyaka Khanepani Limited (KUKL)"},
    {"service": "internet", "keywords": ["internet", "wifi", "worldlink", "subisu"], "provider": "WorldLink Communications"}
]

AUDIO_FILENAME = "nps_live_audio.wav"
RESPONSE_AUDIO = "response_audio.mp3"
SAMPLE_RATE = 16000
CHANNELS = 1

# =====================================================================
# CACHED ASR MODEL LOADING
# =====================================================================
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

with st.spinner("Initializing ASR Core Engine (Faster-Whisper)..."):
    asr_model = load_whisper_model()

# =====================================================================
# HELPER FUNCTIONS (Voice-to-Voice Helpers)
# =====================================================================
def speak_and_play(text):
    """Generates TTS audio and saves it to session state for auto-playing in Streamlit."""
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(RESPONSE_AUDIO)
    st.session_state.last_spoken_text = text
    st.session_state.play_audio_flag = True

def record_audio(duration_seconds=8):
    """Records live audio from microphone."""
    audio_data = sd.rec(
        int(duration_seconds * SAMPLE_RATE), 
        samplerate=SAMPLE_RATE, 
        channels=CHANNELS, 
        dtype='int16'
    )
    sd.wait()
    with wave.open(AUDIO_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

def transcribe_audio():
    """Transcribes audio using Whisper with Nepali/Neplish optimization."""
    segments, info = asr_model.transcribe(
        AUDIO_FILENAME, 
        beam_size=5, 
        vad_filter=True, 
        vad_parameters=dict(min_silence_duration_ms=500),
        language="ne"
    )
    return "".join([segment.text for segment in segments]).strip(), info.language

def parse_voice_command(transcript):
    """NLU Intent & Slot Parser with bilingual keyword matching."""
    t_lower = transcript.lower()
    intent = "unknown"
    if any(k in t_lower for k in ["electricity", "light", "nea", "bijuli", "water", "khanepani", "internet", "wifi", "bill", "pani"]):
        intent = "pay_utility"
    elif any(k in t_lower for k in ["pay", "send", "transfer", "pathauna", "pathaune", "patha", "pathaideu", "pthauna", "dine", "rupee", "rupaiyan", "rs", "rupy"]):
        intent = "transfer_funds"
        
    amount_match = re.search(r'(?:rs\.?|rupees|rupaiyan|rupy)?\s*(\d+)', t_lower)
    amount = int(amount_match.group(1)) if amount_match else None
    
    detected_utility = "Nepal Electricity Authority (NEA)"
    if intent == "pay_utility":
        for utility in UTILITY_PROVIDERS:
            if any(kw in t_lower for kw in utility["keywords"]):
                detected_utility = utility["provider"]
                break
                
    recipient_query = ""
    if intent == "transfer_funds":
        cleaned = t_lower
        stop_words = ["pay", "send", "transfer", "rs", "rupees", "rupaiyan", "rupy", "to", "laai", "lai", "pathauna", "pathaune", "patha", "pathaideu", "pthauna", "ma", "le", "nepal"]
        for word in stop_words:
            cleaned = cleaned.replace(word, "")
        if amount:
            cleaned = cleaned.replace(str(amount), "")
        recipient_query = cleaned.strip().title()

    return {"intent": intent, "amount": amount, "recipient_query": recipient_query, "utility_service": detected_utility}

def resolve_recipient(query):
    """Fuzzy matching contact resolution."""
    contact_names = [c["name"] for c in USER_CONTACTS]
    matches = process.extract(query, contact_names, scorer=fuzz.token_sort_ratio, limit=3)
    valid_matches = [m for m in matches if m[1] > 30]
    
    if not valid_matches:
        return None, "no_match"
        
    matched_contacts = []
    top_score = valid_matches[0][1]
    for match in valid_matches:
        if match[1] >= top_score - 25:
            for c in USER_CONTACTS:
                if c["name"].lower() == match[0].lower() and c not in matched_contacts:
                    matched_contacts.append(c)
                    
    if len(matched_contacts) > 1:
        return matched_contacts, "ambiguous"
        
    return matched_contacts[0], "exact"

# =====================================================================
# STREAMLIT UI LAYOUT
# =====================================================================
st.title("🇳🇵 Nepal Payment Solution (NPS)")
st.subheader("Voice-to-Voice Interactive Utility & Payment Hub")
st.markdown("---")

# Initialize Session State
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "parsed" not in st.session_state:
    st.session_state.parsed = None
if "step" not in st.session_state:
    st.session_state.step = "idle"
if "play_audio_flag" not in st.session_state:
    st.session_state.play_audio_flag = False
if "last_spoken_text" not in st.session_state:
    st.session_state.last_spoken_text = ""

# Audio playback widget if assistant spoke
if st.session_state.play_audio_flag and os.path.exists(RESPONSE_AUDIO):
    st.audio(RESPONSE_AUDIO, format="audio/mp3", autoplay=True)
    st.info(f"🔊 **Assistant Spoke:** \"{st.session_state.last_spoken_text}\"")

# Main Action Button
col1, col2 = st.columns([1, 1])
with col1:
    record_button = st.button("🎤 Click & Speak Command", type="primary", use_container_width=True)

if record_button:
    st.session_state.play_audio_flag = False
    with st.spinner("Listening to voice command (8 seconds)..."):
        record_audio(duration_seconds=8)
    with st.spinner("Processing speech transcription & NLU..."):
        transcript, lang = transcribe_audio()
        st.session_state.transcript = transcript
        st.session_state.parsed = parse_voice_command(transcript)
        st.session_state.step = "process"

# Display Transcript
if st.session_state.transcript:
    st.markdown("### 📝 Live ASR & NLU Analysis")
    st.write(f"**Heard:** \"{st.session_state.transcript}\"")
    st.json(st.session_state.parsed)

# Voice-to-Voice Interactive Logic Loop
if st.session_state.step == "process" and st.session_state.parsed:
    parsed = st.session_state.parsed
    
    if parsed["intent"] == "unknown":
        msg = "I could not understand your transaction request. Please try stating it again."
        speak_and_play(msg)
        st.session_state.step = "idle"
        st.rerun()
        
    elif parsed["intent"] == "transfer_funds":
        # Check missing amount
        if not parsed["amount"]:
            msg = "How much money would you like to transfer? Please specify the amount."
            speak_and_play(msg)
            st.error("❌ Missing Amount in command.")
        # Check missing recipient
        elif not parsed["recipient_query"]:
            msg = "Who would you like to send money to? Please specify the recipient name."
            speak_and_play(msg)
            st.error("❌ Missing Recipient name.")
        else:
            result, status = resolve_recipient(parsed["recipient_query"])
            
            if status == "no_match":
                msg = f"I could not find a contact matching {parsed['recipient_query']} in your directory."
                speak_and_play(msg)
                st.error(f"❌ {msg}")
            elif status == "ambiguous":
                names = " or ".join([c["name"] for c in result])
                msg = f"I found multiple matching accounts: {names}. Which Ram do you mean?"
                speak_and_play(msg)
                st.warning(f"⚠️ {msg}")
            else:
                recipient = result
                st.session_state.recipient = recipient
                msg = f"Send {parsed['amount']} rupees to {recipient['name']} at {recipient['bank']}? Please type yes to confirm."
                speak_and_play(msg)
                st.session_state.step = "confirm_action"
                st.rerun()

    elif parsed["intent"] == "pay_utility":
        if not parsed["amount"]:
            msg = f"How much is your bill amount for {parsed['utility_service']}?"
            speak_and_play(msg)
            st.error("❌ Missing bill amount.")
        else:
            msg = f"Pay utility bill of {parsed['amount']} rupees for {parsed['utility_service']}? Please confirm."
            speak_and_play(msg)
            st.session_state.step = "confirm_action"
            st.rerun()

# Confirmation & OTP Step
if st.session_state.step == "confirm_action":
    st.markdown("### 🔒 Security & Confirmation")
    
    if st.button("Confirm Transaction"):
        otp = random.randint(100000, 999999)
        st.session_state.otp_code = otp
        msg = f"Security authorization required. Your 6-digit code is {otp}."
        speak_and_play(msg)
        st.session_state.step = "otp_entry"
        st.rerun()

if st.session_state.step == "otp_entry":
    st.markdown("### 🔐 Enter OTP Code")
    st.info(f"Spoken OTP Code: **{st.session_state.otp_code}**")
    user_otp = st.text_input("Enter 6-digit code:", max_chars=6)
    
    if st.button("Verify OTP"):
        if user_otp == str(st.session_state.otp_code):
            success_msg = "Transaction successfully completed and logged to ledger."
            speak_and_play(success_msg)
            st.session_state.step = "success"
            st.rerun()
        else:
            st.error("Incorrect OTP.")

if st.session_state.step == "success":
    st.balloons()
    st.success("🎉 Transaction Completed Successfully!")
    if st.button("Reset Hub"):
        st.session_state.transcript = ""
        st.session_state.parsed = None
        st.session_state.step = "idle"
        st.session_state.play_audio_flag = False
        st.rerun()