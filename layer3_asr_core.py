import os
import wave
import re
import random
import sounddevice as sd
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz
from gtts import gTTS

# =====================================================================
# CONFIGURATION & MOCK DATABASES (Nepal Payment Solution - NPS)
# =====================================================================
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_FILENAME = "nps_live_audio.wav"

# Mock Database of P2P Contacts (Notice multiple Rams for disambiguation testing)
USER_CONTACTS = [
    {"id": "ACC101", "name": "Ram Sharma", "bank": "Nabil Bank"},
    {"id": "ACC102", "name": "Ram Kumar Thapa", "bank": "NIC Asia"},
    {"id": "ACC103", "name": "Ramesh Karki", "bank": "Global IME"},
    {"id": "ACC104", "name": "Sita Devi", "bank": "eSewa Wallet"}
]

# Mock Utility Providers
UTILITY_PROVIDERS = [
    {"service": "electricity", "keywords": ["electricity", "light", "nea", "buli", "bijuli"], "provider": "Nepal Electricity Authority (NEA)"},
    {"service": "water", "keywords": ["water", "khanepani", "pani"], "provider": "Kathmandu Upatyaka Khanepani Limited (KUKL)"},
    {"service": "internet", "keywords": ["internet", "wifi", "worldlink", "subisu"], "provider": "WorldLink Communications"}
]

print("=" * 65)
print("   NEPAL PAYMENT SOLUTION (NPS) - FULL VOICE-TO-VOICE HUB   ")
print("=" * 65)

# Load Faster-Whisper Model (Using 'base' model locally on CPU)
print(" Loading Faster-Whisper ASR model...")
asr_model = WhisperModel("base", device="cpu", compute_type="int8")
print(" Whisper model loaded successfully!\n")


# =====================================================================
# STEP 1: TEXT-TO-SPEECH (TTS) WITH AUTO-PLAY (Mac Optimized)
# =====================================================================
def speak_response(text):
    """
    Converts text response into spoken audio and plays it automatically 
    using macOS native 'afplay' command.
    """
    print(f"\n[Voice Assistant 🗣️]: \"{text}\"")
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        # Play audio automatically on macOS
        os.system("afplay response.mp3")
    except Exception as e:
        print(f"[TTS Warning]: {e}")


# =====================================================================
# STEP 2: MICROPHONE AUDIO RECORDING
# =====================================================================
def record_user_audio(duration_seconds=8):
    """Records live voice from the computer microphone."""
    print(f"\n [LISTENING LIVE]: Speak clearly into your microphone now ({duration_seconds} seconds)...")
    print("    Try saying: 'Ram lai 1000 rupey patha' or 'Pay electricity bill of 1500'")
    
    audio_data = sd.rec(
        int(duration_seconds * SAMPLE_RATE), 
        samplerate=SAMPLE_RATE, 
        channels=CHANNELS, 
        dtype='int16'
    )
    sd.wait() # Wait until recording finishes
    print(" [RECORDING STOPPED]: Processing audio...")
    
    # Save to WAV format
    with wave.open(AUDIO_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())


# =====================================================================
# STEP 3: ASR TRANSCRIPTION (Optimized for Nepali & Neplish Code-Switching)
# =====================================================================
def transcribe_audio():
    """Transcribes audio file using Whisper with language optimization."""
    print("⚙️ [ASR RUNNING]: Converting your voice to text...")
    
    # language="ne" helps Whisper accurately decode Nepali phonetics and Neplish
    segments, info = asr_model.transcribe(
        AUDIO_FILENAME, 
        beam_size=5, 
        vad_filter=True, # Removes background room noise/silence
        vad_parameters=dict(min_silence_duration_ms=500),
        language="ne"
    )
    
    transcript = "".join([segment.text for segment in segments]).strip()
    
    print("\n" + "-" * 50)
    print(f" [DEBUG - WHAT WHISPER HEARD]: \"{transcript}\"")
    print(f" [DEBUG - DETECTED LANGUAGE]: {info.language} (Confidence: {info.language_probability:.2f})")
    print("-" * 50 + "\n")
    
    return transcript


# =====================================================================
# STEP 4: INTENT & SLOT PARSER (Bilingual Neplish Support)
# =====================================================================
def parse_voice_command(transcript):
    """Normalizes bilingual commands, amounts, verbs, and targets."""
    t_lower = transcript.lower()
    
    # 1. Determine Intent (Utility vs P2P Transfer)
    intent = "unknown"
    if any(k in t_lower for k in ["electricity", "light", "nea", "bijuli", "water", "khanepani", "internet", "wifi", "bill", "pani"]):
        intent = "pay_utility"
    elif any(k in t_lower for k in ["pay", "send", "transfer", "pathauna", "pathaune", "patha", "pathaideu", "pthauna", "dine", "rupee", "rupaiyan", "rs", "rupy"]):
        intent = "transfer_funds"
        
    # 2. Extract Numerical Amount
    amount_match = re.search(r'(?:rs\.?|rupees|rupaiyan|rupy)?\s*(\d+)', t_lower)
    amount = int(amount_match.group(1)) if amount_match else None
    
    # 3. Extract Utility Provider
    detected_utility = "Nepal Electricity Authority (NEA)"
    if intent == "pay_utility":
        for utility in UTILITY_PROVIDERS:
            if any(kw in t_lower for kw in utility["keywords"]):
                detected_utility = utility["provider"]
                break
                
    # 4. Extract P2P Recipient Name (Stripping Nepali & English stop words)
    recipient_query = ""
    if intent == "transfer_funds":
        cleaned = t_lower
        stop_words = ["pay", "send", "transfer", "rs", "rupees", "rupaiyan", "rupy", "to", "laai", "lai", "pathauna", "pathaune", "patha", "pathaideu", "pthauna", "ma", "le", "nepal"]
        for word in stop_words:
            cleaned = cleaned.replace(word, "")
        if amount:
            cleaned = cleaned.replace(str(amount), "")
        recipient_query = cleaned.strip().title()

    return {
        "intent": intent,
        "amount": amount,
        "recipient_query": recipient_query,
        "utility_service": detected_utility
    }


# =====================================================================
# STEP 5: CONTACT DISAMBIGUATION (Dynamic Voice Questioning)
# =====================================================================
def resolve_recipient(query):
    """Fuzzy matches name queries against core banking records."""
    contact_names = [c["name"] for c in USER_CONTACTS]
    matches = process.extract(query, contact_names, scorer=fuzz.token_sort_ratio, limit=3)
    valid_matches = [m for m in matches if m[1] > 30] # Flexible threshold for Neplish
    
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
# STEP 6: FULL VOICE-TO-VOICE WORKFLOW EXECUTION
# =====================================================================
def main():
    speak_response("Nepal payment solution voice hub is ready. Please state your transaction command.")
    input("Press [Enter] on your keyboard, wait for the listening prompt, then speak...")
    
    # 1. Record User Voice
    record_user_audio(duration_seconds=8)
    
    # 2. Transcribe Audio via ASR
    transcript = transcribe_audio()
    if not transcript:
        speak_response("I couldn't hear anything. Please try again.")
        return

    # 3. Parse Intent and Slots
    parsed = parse_voice_command(transcript)
    print(f" [PARSED DATA OBJECT]: {parsed}\n")
    
    if parsed["intent"] == "unknown" or not parsed["amount"]:
        speak_response("I couldn't understand the amount or transaction type. Please try again.")
        return

    # --- HANDLE P2P TRANSFERS ---
    if parsed["intent"] == "transfer_funds":
        if not parsed["recipient_query"]:
            speak_response("Please specify who you are sending money to.")
            return
            
        result, status = resolve_recipient(parsed["recipient_query"])
        
        if status == "no_match":
            speak_response(f"I couldn't find a contact matching {parsed['recipient_query']}.")
            return
            
        elif status == "ambiguous":
            # Dynamic Voice Questioning for Ambiguity!
            names_spoken = " or ".join([c["name"] for c in result])
            ambig_msg = f"I found multiple accounts matching your request: {names_spoken}. Which one do you mean?"
            speak_response(ambig_msg)
            
            print(f"\n⚠️ [AMBIGUITY DETECTED]: Spoke options aloud -> {names_spoken}")
            print("Listening for clarification...")
            
            # Record clarification speech
            record_user_audio(duration_seconds=6)
            clarification_transcript = transcribe_audio()
            
            # Re-resolve with clarification
            clarified_result, clarified_status = resolve_recipient(clarification_transcript)
            if clarified_status == "exact":
                recipient = clarified_result
            else:
                speak_response("Could not resolve specific contact. Transaction aborted.")
                return
        else:
            recipient = result
        
        # Confirmation & OTP Security Loop
        confirm_msg = f"Send {parsed['amount']} rupees to {recipient['name']} at {recipient['bank']}? Please say yes or type it in terminal to confirm."
        speak_response(confirm_msg)
        
        confirm = input(f"\n{confirm_msg} (Type 'yes' to confirm): ").strip().lower()
        if confirm in ["yes", "y"]:
            otp = random.randint(100000, 999999)
            speak_response(f"Security verification required. Your 6-digit confirmation code is {otp}.")
            
            code = input(f" Enter 6-digit code ({otp}): ").strip()
            if code == str(otp):
                success_msg = f"Success! Transferred {parsed['amount']} rupees to {recipient['name']}. Receipt generated."
                speak_response(success_msg)
                print(" [TRANSACTION SUCCESSFUL]")
            else:
                speak_response("Incorrect code entered. Transaction aborted.")
        else:
            speak_response("Transaction cancelled.")

    # --- HANDLE UTILITY BILLS ---
    elif parsed["intent"] == "pay_utility":
        utility_msg = f"Pay utility bill of {parsed['amount']} rupees for {parsed['utility_service']}? Confirm by saying yes."
        speak_response(utility_msg)
        
        confirm = input(f"\n{utility_msg} (Type 'yes' to confirm): ").strip().lower()
        if confirm in ["yes", "y"]:
            otp = random.randint(100000, 999999)
            speak_response(f"Utility security code is {otp}.")
            
            code = input(f" Enter 6-digit code ({otp}): ").strip()
            if code == str(otp):
                success_msg = f"Successfully paid {parsed['amount']} rupees to {parsed['utility_service']}. Receipt generated."
                speak_response(success_msg)
                print(" [UTILITY PAID SUCCESSFULLY]")
            else:
                speak_response("Incorrect code entered. Payment aborted.")
        else:
            speak_response("Utility payment cancelled.")

if __name__ == "__main__":
    main()