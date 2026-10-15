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

print("=" * 65)
print("   NEPAL PAYMENT SOLUTION (NPS) - CONVERSATIONAL VOICE HUB   ")
print("=" * 65)

print(" Loading Faster-Whisper ASR model...")
asr_model = WhisperModel("base", device="cpu", compute_type="int8")
print(" Whisper model loaded successfully!\n")


def speak_response(text):
    print(f"\n[Voice Assistant 🗣️]: \"{text}\"")
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        os.system("afplay response.mp3")
    except Exception as e:
        print(f"[TTS Warning]: {e}")


def record_user_audio(duration_seconds=12):  # Increased to 12s for relaxed speaking
    print(f"\n🎤 [LISTENING LIVE]: Speak clearly into your microphone now ({duration_seconds} seconds)...")
    
    audio_data = sd.rec(
        int(duration_seconds * SAMPLE_RATE), 
        samplerate=SAMPLE_RATE, 
        channels=CHANNELS, 
        dtype='int16'
    )
    sd.wait()
    print("🎙️ [RECORDING STOPPED]: Processing audio...")
    
    with wave.open(AUDIO_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())


def transcribe_audio():
    print("⚙️ [ASR RUNNING]: Converting your voice to text...")
    segments, info = asr_model.transcribe(
        AUDIO_FILENAME, 
        beam_size=5, 
        vad_filter=True, 
        vad_parameters=dict(min_silence_duration_ms=500),
        language="ne"
    )
    transcript = "".join([segment.text for segment in segments]).strip()
    print(f" [DEBUG - WHAT WHISPER HEARD]: \"{transcript}\"")
    return transcript


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


def main():
    speak_response("Nepal payment solution voice hub is ready. Please state your transaction command.")
    input("Press [Enter] on your keyboard to start speaking...")
    
    # 1. Initial Recording (12 Seconds)
    record_user_audio(duration_seconds=12)
    transcript = transcribe_audio()
    if not transcript:
        speak_response("I couldn't hear anything. Please try again.")
        return

    parsed = parse_voice_command(transcript)
    print(f"📦 [PARSED DATA OBJECT]: {parsed}\n")
    
    if parsed["intent"] == "unknown":
        speak_response("I didn't catch that as a transfer or bill payment. Please state your request again.")
        return

    # --- HANDLE P2P TRANSFERS ---
    if parsed["intent"] == "transfer_funds":
        # Conversational Recovery: Ask if recipient is missing instead of aborting
        if not parsed["recipient_query"]:
            speak_response("Who would you like to send money to? Please state the recipient name.")
            record_user_audio(duration_seconds=8)
            parsed["recipient_query"] = transcribe_audio().strip().title()

        result, status = resolve_recipient(parsed["recipient_query"])
        
        if status == "no_match":
            speak_response(f"I couldn't find a contact matching {parsed['recipient_query']} in your directory. Transaction aborted.")
            return
            
        elif status == "ambiguous":
            names_spoken = " or ".join([c["name"] for c in result])
            speak_response(f"I found multiple accounts matching your request: {names_spoken}. Which one do you mean?")
            
            record_user_audio(duration_seconds=8)
            clarification_transcript = transcribe_audio()
            
            clarified_result, clarified_status = resolve_recipient(clarification_transcript)
            if clarified_status == "exact":
                recipient = clarified_result
            else:
                speak_response("Could not resolve specific contact. Transaction aborted.")
                return
        else:
            recipient = result

        # Conversational Recovery: Ask if amount is missing instead of aborting
        if not parsed["amount"]:
            speak_response(f"How much money would you like to send to {recipient['name']}?")
            record_user_audio(duration_seconds=8)
            amt_transcript = transcribe_audio()
            amt_match = re.search(r'(\d+)', amt_transcript)
            parsed["amount"] = int(amt_match.group(1)) if amt_match else 1000  # Default fallback
        
        # Voice Confirmation Loop
        confirm_msg = f"Send {parsed['amount']} rupees to {recipient['name']} at {recipient['bank']}? Please say yes to confirm."
        speak_response(confirm_msg)
        
        record_user_audio(duration_seconds=6)
        if any(w in transcribe_audio().lower() for w in ["yes", "yup", "ho", "sure", "ok", "confirm", "sahi"]):
            otp = random.randint(100000, 999999)
            speak_response("Transaction confirmed. Please check your screen to enter your secret code.")
            
            print(f"\n [SECURE MOCK DISPLAY]: Your secret authorization code is: {otp}")
            code = input(f"Enter 6-digit code ({otp}): ").strip()
            
            if code == str(otp):
                speak_response(f"Success! Transferred {parsed['amount']} rupees to {recipient['name']}.")
                print(" [TRANSACTION SUCCESSFUL]")
            else:
                speak_response("Incorrect code entered. Transaction aborted.")
        else:
            speak_response("Confirmation not received. Transaction cancelled.")

    # --- HANDLE UTILITY BILLS ---
    elif parsed["intent"] == "pay_utility":
        # Conversational Recovery: Ask for amount if missing instead of aborting
        if not parsed["amount"]:
            speak_response(f"How much is your bill amount for {parsed['utility_service']}?")
            record_user_audio(duration_seconds=8)
            amt_transcript = transcribe_audio()
            amt_match = re.search(r'(\d+)', amt_transcript)
            parsed["amount"] = int(amt_match.group(1)) if amt_match else 1500

        utility_msg = f"Pay utility bill of {parsed['amount']} rupees for {parsed['utility_service']}? Please say yes to confirm."
        speak_response(utility_msg)
        
        record_user_audio(duration_seconds=6)
        if any(w in transcribe_audio().lower() for w in ["yes", "yup", "ho", "sure", "ok", "confirm", "sahi"]):
            otp = random.randint(100000, 999999)
            speak_response("Utility payment confirmed. Please check your screen for your secret code.")
            
            print(f"\n🔒 [SECURE MOCK DISPLAY]: Your secret utility security code is: {otp}")
            code = input(f"Enter 6-digit code ({otp}): ").strip()
            
            if code == str(otp):
                speak_response(f"Successfully paid {parsed['amount']} rupees to {parsed['utility_service']}.")
                print(" [UTILITY PAID SUCCESSFULLY]")
            else:
                speak_response("Incorrect code entered. Payment aborted.")
        else:
            speak_response("Confirmation not received. Utility payment cancelled.")

if __name__ == "__main__":
    main()
