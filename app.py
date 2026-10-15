import streamlit as st
import os
import wave
import re
import random
import sqlite3
import datetime
import sounddevice as sd
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz
from gtts import gTTS

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Nepal Payment Solution - Executive Voice Hub",
    page_icon="🇳🇵",
    layout="wide"
)

# =====================================================================
# SQLITE DATABASE LEDGER SETUP (Layer 6)
# =====================================================================
DB_FILE = "nps_ledger.db"

def init_db():
    """Initializes the SQLite transaction ledger table if not present."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            tx_id TEXT PRIMARY KEY,
            timestamp TEXT,
            intent TEXT,
            service_or_recipient TEXT,
            account_bank TEXT,
            amount INTEGER,
            status TEXT,
            raw_transcript TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_transaction(tx_id, intent, target, bank, amount, transcript, status="SUCCESS"):
    """Inserts a completed transaction record into the local SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO transactions (tx_id, timestamp, intent, service_or_recipient, account_bank, amount, status, raw_transcript)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (tx_id, timestamp, intent, target, bank, amount, status, transcript))
    conn.commit()
    conn.close()

def fetch_transactions():
    """Retrieves all transaction logs from the SQLite ledger."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

# Initialize DB on load
init_db()

# =====================================================================
# MOCK DATABASES
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
# CACHED ASR MODEL
# =====================================================================
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

with st.spinner("Initializing Local ASR Engine (Faster-Whisper)..."):
    asr_model = load_whisper_model()

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def speak_and_play(text):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(RESPONSE_AUDIO)
    st.session_state.last_spoken_text = text
    st.session_state.play_audio_flag = True

def record_audio(duration_seconds=12):
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
    segments, info = asr_model.transcribe(
        AUDIO_FILENAME, 
        beam_size=5, 
        vad_filter=True, 
        vad_parameters=dict(min_silence_duration_ms=500),
        language="ne"
    )
    return "".join([segment.text for segment in segments]).strip(), info.language

def parse_voice_command(transcript):
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
        stop_words = ["pay", "send", "transfer", "rs", "rupees", "rupaiyan", "rupy", "to", "laai", "lai", "pathauna", "pathaune", "patha", "pathaideu", "pthauna", "ma", "le", "nepal", "paisa"]
        for word in stop_words:
            cleaned = cleaned.replace(word, "")
        if amount:
            cleaned = cleaned.replace(str(amount), "")
        recipient_query = cleaned.strip().title()

    return {"intent": intent, "amount": amount, "recipient_query": recipient_query, "utility_service": detected_utility}

def resolve_recipient(query):
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
# EXECUTIVE DASHBOARD UI (TABBED INTERFACE)
# =====================================================================
st.title("🇳🇵 Nepal Payment Solution (NPS)")
st.subheader("Executive Voice-to-Voice Stakeholder Demonstration Hub")

tab_voice, tab_ledger = st.tabs(["🎙️ Voice Payment Terminal", "📜 SQLite Ledger & Receipt History"])

# Session State Initialization
if "transcript" not in st.session_state: st.session_state.transcript = ""
if "parsed" not in st.session_state: st.session_state.parsed = None
if "step" not in st.session_state: 
    st.session_state.step = "idle"
    speak_and_play("Nepal payment solution voice hub is ready. Please state your transaction command.")

if "play_audio_flag" not in st.session_state: st.session_state.play_audio_flag = False
if "last_spoken_text" not in st.session_state: st.session_state.last_spoken_text = ""

# Sidebar Progress Tracking
st.sidebar.title(" Pipeline Progress")
steps_status = {
    "idle": "1. Ready / Initialized",
    "process": "2. Intent & Slot Filling",
    "ask_recipient": "3. Collecting Recipient",
    "ask_amount": "4. Collecting Amount",
    "confirm_voice": "5. Voice Confirmation",
    "otp_stage": "6. Secure OTP Verification",
    "success": "7. Ledger Commit Complete"
}
for k, v in steps_status.items():
    if st.session_state.step == k:
        st.sidebar.markdown(f"👉 **{v}**")
    else:
        st.sidebar.markdown(f"<span style='color:grey;'>{v}</span>", unsafe_allow_html=True)


# =====================================================================
# TAB 1: VOICE PAYMENT TERMINAL
# =====================================================================
with tab_voice:
    col_ctrl, col_telemetry = st.columns([1.2, 1])

    with col_ctrl:
        st.markdown("### 🎙️ Live Voice Control")
        st.info("The assistant is ready. Click below to record your voice command or respond to prompts.")
        
        btn_label = "🎤 Start Voice Command (12s)"
        if st.session_state.step == "ask_recipient":
            btn_label = "🎤 Speak Recipient Name (12s)"
        elif st.session_state.step == "ask_amount":
            btn_label = "🎤 Speak Amount (12s)"
        elif st.session_state.step == "confirm_voice":
            btn_label = "🎤 Speak Confirmation (Yes/No)"

        if st.button(btn_label, type="primary", use_container_width=True):
            st.session_state.play_audio_flag = False
            with st.spinner("Recording live audio (12 seconds)... Speak clearly!"):
                record_audio(duration_seconds=12)
            
            transcript, _ = transcribe_audio()
            st.session_state.transcript = transcript

            if st.session_state.step == "idle":
                st.session_state.parsed = parse_voice_command(transcript)
                st.session_state.step = "process"
                st.rerun()
                
            elif st.session_state.step == "ask_recipient":
                st.session_state.parsed["recipient_query"] = transcript.strip().title()
                st.session_state.step = "process"
                st.rerun()
                
            elif st.session_state.step == "ask_amount":
                amt_match = re.search(r'(\d+)', transcript)
                if amt_match:
                    st.session_state.parsed["amount"] = int(amt_match.group(1))
                st.session_state.step = "process"
                st.rerun()
                
            elif st.session_state.step == "confirm_voice":
                t_conf = transcript.lower()
                if any(w in t_conf for w in ["yes", "yup", "ho", "sure", "ok", "confirm", "sahi"]):
                    otp = random.randint(100000, 999999)
                    st.session_state.otp_code = otp
                    speak_and_play("Transaction confirmed. Please check your screen to enter your secret authorization code.")
                    st.session_state.step = "otp_stage"
                    st.rerun()
                elif any(w in t_conf for w in ["no", "nah", "haina", "cancel", "stop"]):
                    speak_and_play("Transaction cancelled by user voice command. Resetting hub.")
                    st.session_state.step = "idle"
                    st.rerun()
                else:
                    speak_and_play("I did not catch that clearly. Please say yes or no.")
                    st.rerun()

        # Audio Feedback Player
        if st.session_state.play_audio_flag and os.path.exists(RESPONSE_AUDIO):
            st.audio(RESPONSE_AUDIO, format="audio/mp3", autoplay=True)
            st.success(f" **Assistant Spoke:** \"{st.session_state.last_spoken_text}\"")

    with col_telemetry:
        st.markdown("###  System Telemetry & NLU State")
        if st.session_state.transcript:
            st.text_area("Live Transcript Output", st.session_state.transcript, height=70)
            st.json(st.session_state.parsed)
        else:
            st.info("System standing by for turn-by-turn voice dialog.")

    st.markdown("---")

    # Workflow Execution Logic
    if st.session_state.step == "process" and st.session_state.parsed:
        parsed = st.session_state.parsed
        
        if parsed["intent"] == "unknown":
            speak_and_play("I did not recognize that command. Please state a fund transfer or utility payment.")
            st.session_state.step = "idle"
            st.rerun()
            
        elif parsed["intent"] == "transfer_funds":
            if not parsed["recipient_query"]:
                msg = "Who would you like to send money to? Please state the recipient name."
                speak_and_play(msg)
                st.session_state.step = "ask_recipient"
                st.rerun()
            else:
                result, status = resolve_recipient(parsed["recipient_query"])
                if status == "no_match":
                    msg = f"Contact {parsed['recipient_query']} not found in directory. Transaction aborted."
                    speak_and_play(msg)
                    st.session_state.step = "idle"
                    st.rerun()
                elif status == "ambiguous":
                    names = " or ".join([c["name"] for c in result])
                    msg = f"Multiple accounts found: {names}. Which one do you mean?"
                    speak_and_play(msg)
                    st.session_state.step = "ask_recipient"
                    st.rerun()
                else:
                    recipient = result
                    st.session_state.recipient = recipient
                    
                    if not parsed["amount"]:
                        msg = f"How much money would you like to send to {recipient['name']}?"
                        speak_and_play(msg)
                        st.session_state.step = "ask_amount"
                        st.rerun()
                    else:
                        msg = f"Send {parsed['amount']} rupees to {recipient['name']} at {recipient['bank']}? Please say yes to confirm, or no to cancel."
                        speak_and_play(msg)
                        st.session_state.step = "confirm_voice"
                        st.rerun()

        elif parsed["intent"] == "pay_utility":
            if not parsed["amount"]:
                msg = f"How much is your bill amount for {parsed['utility_service']}?"
                speak_and_play(msg)
                st.session_state.step = "ask_amount"
                st.rerun()
            else:
                msg = f"Pay utility bill of {parsed['amount']} rupees for {parsed['utility_service']}? Please say yes to confirm, or no to cancel."
                speak_and_play(msg)
                st.session_state.step = "confirm_voice"
                st.rerun()

    # OTP Verification Stage
    if st.session_state.step == "otp_stage":
        st.markdown("###  Secure Multi-Factor Authentication (OTP)")
        st.warning(" [SECURITY PROTOCOL]: OTP is hidden from voice output and displayed securely on screen.")
        st.info(f"**Generated Secret OTP:** {st.session_state.otp_code}")
        
        user_otp = st.text_input("Enter 6-digit verification code:", max_chars=6)
        
        if st.button("Verify & Commit Transaction"):
            if user_otp == str(st.session_state.otp_code):
                # GENERATE TX ID & COMMIT TO SQLITE LEDGER
                tx_id = f"NPS-{random.randint(10000000, 99999999)}"
                st.session_state.last_tx_id = tx_id
                
                parsed = st.session_state.parsed
                if parsed["intent"] == "transfer_funds":
                    target = st.session_state.recipient["name"]
                    bank = st.session_state.recipient["bank"]
                else:
                    target = parsed["utility_service"]
                    bank = "NPS Utility Gateway"
                    
                log_transaction(
                    tx_id=tx_id,
                    intent=parsed["intent"],
                    target=target,
                    bank=bank,
                    amount=parsed["amount"],
                    transcript=st.session_state.transcript
                )
                
                speak_and_play("Transaction successfully completed, verified, and logged to ledger.")
                st.session_state.step = "success"
                st.rerun()
            else:
                st.error("Incorrect verification code.")

    # Success Stage
    if st.session_state.step == "success":
        st.balloons()
        st.success(f"🎉 Transaction Successfully Committed to SQLite Ledger! Reference ID: `{st.session_state.get('last_tx_id', 'N/A')}`")
        if st.button("Start New Voice Transaction"):
            st.session_state.transcript = ""
            st.session_state.parsed = None
            st.session_state.step = "idle"
            st.session_state.play_audio_flag = False
            st.rerun()


# =====================================================================
# TAB 2: SQLITE LEDGER & RECEIPT HISTORY
# =====================================================================
with tab_ledger:
    st.markdown("###  Permanent Transaction History (SQLite Database)")
    st.info("Every voice-verified transaction is permanently stored in local database `nps_ledger.db`.")
    
    rows = fetch_transactions()
    
    if not rows:
        st.warning("No transactions recorded in ledger yet. Complete a voice payment to view records.")
    else:
        # Display Summary Table
        table_data = []
        for r in rows:
            table_data.append({
                "TX Reference ID": r[0],
                "Timestamp": r[1],
                "Intent": r[2].upper(),
                "Recipient / Utility": r[3],
                "Financial Provider": r[4],
                "Amount (NPR)": f"Rs. {r[5]:,}",
                "Status": r[6]
            })
        st.dataframe(table_data, use_container_width=True)
        
        st.markdown("---")
        st.markdown("###  Interactive Receipt Generator")
        selected_tx = st.selectbox("Select Transaction ID to view formal receipt:", [r[0] for r in rows])
        
        # Display Printable Audit Receipt Card
        if selected_tx:
            tx_detail = next(r for r in rows if r[0] == selected_tx)
            
            receipt_box = f"""
            ```text
            =============================================================
                           NEPAL PAYMENT SOLUTION (NPS)
                            OFFICIAL TRANSACTION RECEIPT
            =============================================================
            TRANSACTION ID : {tx_detail[0]}
            DATE & TIME    : {tx_detail[1]}
            STATUS         : {tx_detail[6]}
            
            TYPE           : {tx_detail[2].upper()}
            BENEFICIARY    : {tx_detail[3]}
            INSTITUTION    : {tx_detail[4]}
            AMOUNT PAID    : Rs. {tx_detail[5]:,}.00 NPR
            
            VOICE COMMAND  : "{tx_detail[7]}"
            VERIFICATION   : 6-Digit Voice-Triggered OTP Verified
            =============================================================
                          Thank you for using NPS Voice Hub!
            ```
            """
            st.markdown(receipt_box)
            st.download_button(
                label="📄 Download Official Receipt (.txt)",
                data=receipt_box.replace("```text\n", "").replace("\n            ```", ""),
                file_name=f"Receipt_{selected_tx}.txt",
                mime="text/plain"
            )