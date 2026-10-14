import os
import wave
import re
import random
import numpy as np
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

# Mock Database of P2P Contacts
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
print("   NEPAL PAYMENT SOLUTION (NPS) - VOICE ASSISTANT DEBUGGER   ")
print("=" * 65)

# Load Faster-Whisper Model (Using 'base' model locally on CPU)
print(" Loading Faster-Whisper ASR model...")
asr_model = WhisperModel("base", device="cpu", compute_type="int8")
print(" Whisper model loaded successfully!\n")


# =====================================================================
# STEP 1: TEXT-TO-SPEECH (TTS) AUDIO FEEDBACK
# =====================================================================
def speak_response(text):
    """Speaks out confirmation messages to the user."""
    print(f"\n[Voice Assistant 🗣️]: \"{text}\"")
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
    except Exception as e:
        print(f"[TTS Warning]: {e}")


# =====================================================================
# STEP 2: MICROPHONE AUDIO RECORDING (12 Seconds Window)
# =====================================================================
def record_user_audio(duration_seconds=12):
    """Records voice from the computer microphone safely."""
    print(f"\n [LISTENING LIVE]: Speak clearly into your microphone now ({duration_seconds} seconds)...")
    print("    Try saying: 'Send 5000 rupees to Ram' or 'Pay electricity bill'")
    
    audio_data = sd.rec(
        int(duration_seconds * SAMPLE_RATE), 
        samplerate=SAMPLE_RATE, 
        channels=CHANNELS, 
        dtype='int16'
    )
    sd.wait() # Wait until recording finishes
    print("🎙️ [RECORDING STOPPED]: Saving audio file...")
    
    # Save to WAV format
    with wave.open(AUDIO_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())


# =====================================================================
# STEP 3: ASR TRANSCRIPTION (WITH DEBUG LOGGING)
# =====================================================================
def transcribe_audio():
    """Transcribes audio file and prints raw output for debugging."""
    print("⚙️ [ASR RUNNING]: Converting your voice to text...")
    
    # We pass language=None to allow both English and Nepali/Neplish code-switching
    segments, info = asr_model.transcribe(
        AUDIO_FILENAME, 
        beam_size=5, 
        vad_filter=True, # Removes background room noise/silence
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    transcript = "".join([segment.text for segment in segments]).strip()
    
    print("\n" + "=" * 50)
    print(f" [DEBUG - WHAT WHISPER HEARD]: \"{transcript}\"")
    print(f" [DEBUG - DETECTED LANGUAGE]: {info.language} (Confidence: {info.language_probability:.2f})")
    print("=" * 50 + "\n")
    
    return transcript


# =====================================================================
# STEP 4: INTENT & SLOT PARSER (FLEXIBLE SYNONYMS)
# =====================================================================
def parse_voice_command(transcript):
    """Normalizes commands, amounts, verbs, and targets."""
    t_lower = transcript.lower()
    
    # 1. Determine Intent (Utility vs P2P Transfer)
    intent = "unknown"
    if any(k in t_lower for k in ["electricity", "light", "nea", "bijuli", "water", "khanepani", "internet", "wifi", "bill"]):
        intent = "pay_utility"
    elif any(k in t_lower for k in ["pay", "send", "transfer", "pathauna", "pathaune", "pthauna", "dine", "rupee", "rs"]):
        intent = "transfer_funds"
        
    # 2. Extract Numerical Amount
    amount_match = re.search(r'(?:rs\.?|rupees|rupaiyan)?\s*(\d+)', t_lower)
    amount = int(amount_match.group(1)) if amount_match else None
    
    # 3. Extract Utility Provider
    detected_utility = "Nepal Electricity Authority (NEA)"
    if intent == "pay_utility":
        for utility in UTILITY_PROVIDERS:
            if any(kw in t_lower for kw in utility["keywords"]):
                detected_utility = utility["provider"]
                break
                
    # 4. Extract P2P Recipient Name
    recipient_query = ""
    if intent == "transfer_funds":
        cleaned = t_lower
        # Strip out action verbs and filler words
        stop_words = ["pay", "send", "transfer", "rs", "rupees", "rupaiyan", "to", "laai", "lai", "pathauna", "pathaune", "pthauna", "ma", "le"]
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
# STEP 5: CONTACT DISAMBIGUATION (RAM SHARMA vs RAM KUMAR THAPA)
# =====================================================================
def resolve_recipient(query):
    """Fuzzy matches name queries against core banking records."""
    contact_names = [c["name"] for c in USER_CONTACTS]
    matches = process.extract(query, contact_names, scorer=fuzz.token_sort_ratio, limit=3)
    valid_matches = [m for m in matches if m[1] > 35] # Lowered threshold slightly for safety
    
    if not valid_matches:
        return None, "no_match"
        
    matched_contacts = []
    top_score = valid_matches[0][1]
    for match in valid_matches:
        if match[1] >= top_score - 20:
            for c in USER_CONTACTS:
                if c["name"].lower() == match[0].lower() and c not in matched_contacts:
                    matched_contacts.append(c)
                    
    if len(matched_contacts) > 1:
        return matched_contacts, "ambiguous"
        
    return matched_contacts[0], "exact"


# =====================================================================
# STEP 6: SECURE WORKFLOW EXECUTION
# =====================================================================
def main():
    input("Press [Enter] on your keyboard, wait for the listening prompt, then speak...")
    
    # Record from Mic
    record_user_audio(duration_seconds=12)
    
    # Transcribe audio
    transcript = transcribe_audio()
    if not transcript:
        speak_response("I couldn't hear anything. Please check your microphone and try again.")
        return

    # Parse intent and slots
    parsed = parse_voice_command(transcript)
    print(f" [PARSED DATA OBJECT]: {parsed}\n")
    
    if parsed["intent"] == "unknown" or not parsed["amount"]:
        speak_response("I couldn't understand the amount or transaction type. Please try speaking clearly again.")
        return

    # Handle P2P Transfers
    if parsed["intent"] == "transfer_funds":
        if not parsed["recipient_query"]:
            speak_response("Please specify who you are sending money to.")
            return
            
        result, status = resolve_recipient(parsed["recipient_query"])
        
        if status == "no_match":
            speak_response(f"I couldn't find a contact matching {parsed['recipient_query']}.")
            return
        elif status == "ambiguous":
            names = " or ".join([c["name"] for c in result])
            speak_response(f"I found multiple accounts: {names}. Which one do you mean?")
            return
            
        recipient = result
        msg = f"Send {parsed['amount']} rupees to {recipient['name']} at {recipient['bank']}?"
        speak_response(msg)
        
        confirm = input(f"\n{msg} (Type 'yes' to confirm): ").strip().lower()
        if confirm in ["yes", "y"]:
            otp = random.randint(100000, 999999)
            speak_response(f"Your secure confirmation code is {otp}.")
            code = input(f"Enter 6-digit code ({otp}): ").strip()
            if code == str(otp):
                speak_response(f"Success! Transferred {parsed['amount']} rupees to {recipient['name']}.")
                print(" [TRANSACTION SUCCESSFUL]")
            else:
                speak_response("Incorrect code. Aborted.")
        else:
            speak_response("Cancelled.")

    # Handle Utility Bills
    elif parsed["intent"] == "pay_utility":
        msg = f"Pay utility bill of {parsed['amount']} rupees for {parsed['utility_service']}?"
        speak_response(msg)
        
        confirm = input(f"\n{msg} (Type 'yes' to confirm): ").strip().lower()
        if confirm in ["yes", "y"]:
            otp = random.randint(100000, 999999)
            speak_response(f"Utility security code is {otp}.")
            code = input(f"Enter 6-digit code ({otp}): ").strip()
            if code == str(otp):
                speak_response(f"Successfully paid {parsed['amount']} rupees. Receipt generated.")
                print(" [UTILITY PAID SUCCESSFULLY]")
            else:
                speak_response("Incorrect code. Aborted.")
        else:
            speak_response("Cancelled.")

if __name__ == "__main__":
    main()