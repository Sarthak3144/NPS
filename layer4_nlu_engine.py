import re
from rapidfuzz import process, fuzz

# =====================================================================
# LAYER 4: ADVANCED NLU & SLOT EXTRACTION ENGINE (FOR NPS)
# =====================================================================

# Mock Reference Databases for NLU Grounding
SUPPORTED_CONTACTS = [
    {"id": "ACC101", "name": "Ram Sharma", "bank": "Nabil Bank"},
    {"id": "ACC102", "name": "Ram Kumar Thapa", "bank": "NIC Asia"},
    {"id": "ACC103", "name": "Ramesh Karki", "bank": "Global IME"},
    {"id": "ACC104", "name": "Sita Devi", "bank": "eSewa Wallet"}
]

UTILITY_DICTIONARY = [
    {"service": "electricity", "keywords": ["electricity", "light", "nea", "bijuli", "buli", "bhatti"], "provider": "Nepal Electricity Authority (NEA)"},
    {"service": "water", "keywords": ["water", "khanepani", "pani", "dhara"], "provider": "Kathmandu Upatyaka Khanepani Limited (KUKL)"},
    {"service": "internet", "keywords": ["internet", "wifi", "worldlink", "subisu", "net", "broadband"], "provider": "WorldLink Communications"}
]


class NPSNluEngine:
    """
    Advanced Natural Language Understanding engine for Nepal Payment Solution.
    Transforms raw ASR transcripts into strict structured JSON entities.
    """
    
    def __init__(self):
        print("🚀 [Layer 4 Initialized]: NLU Engine ready for slot extraction.")

    def classify_intent(self, transcript: str) -> str:
        """Determines whether the user wants a fund transfer or utility payment."""
        t_lower = transcript.lower()
        
        # Check for utility payment triggers (English + Nepali/Neplish)
        utility_triggers = ["electricity", "light", "nea", "bijuli", "water", "khanepani", "pani", "internet", "wifi", "bill", "broadband"]
        if any(keyword in t_lower for keyword in utility_triggers):
            return "pay_utility"
            
        # Check for P2P fund transfer triggers
        transfer_triggers = ["pay", "send", "transfer", "pathauna", "pathaune", "pathaideu", "pathau", "dine", "rupee", "rupaiyan", "rs"]
        if any(keyword in t_lower for keyword in transfer_triggers):
            return "transfer_funds"
            
        return "unknown"

    def extract_amount(self, transcript: str):
        """Extracts numerical financial values from unstructured text."""
        # Searches for numbers preceded or followed by currency markers, or standalone integers
        amount_match = re.search(r'(?:rs\.?|rupees|rupaiyan)?\s*(\d+)', transcript.lower())
        return int(amount_match.group(1)) if amount_match else None

    def extract_utility_provider(self, transcript: str) -> str:
        """Maps utility keywords to formal institutional service providers."""
        t_lower = transcript.lower()
        for utility in UTILITY_DICTIONARY:
            if any(kw in t_lower for kw in utility["keywords"]):
                return utility["provider"]
        return "Nepal Electricity Authority (NEA)" # Default fallback

    def extract_recipient_and_resolve(self, transcript: str, amount):
        """
        Cleans recipient name queries and performs fuzzy matching 
        against core banking user directories.
        """
        t_lower = transcript.lower()
        cleaned = t_lower
        
        # Comprehensive stop-word stripping for Neplish & English
        stop_words = ["pay", "send", "transfer", "rs", "rupees", "rupaiyan", "to", "laai", "lai", "pathauna", "pathaune", "pathaideu", "pathau", "ma", "le", "nepal", "bill", "for"]
        for word in stop_words:
            cleaned = cleaned.replace(word, "")
            
        if amount:
            cleaned = cleaned.replace(str(amount), "")
            
        recipient_query = cleaned.strip().title()
        if not recipient_query:
            return None, "no_query"

        # Fuzzy Matching Resolution via RapidFuzz
        contact_names = [c["name"] for c in SUPPORTED_CONTACTS]
        matches = process.extract(recipient_query, contact_names, scorer=fuzz.token_sort_ratio, limit=3)
        valid_matches = [m for m in matches if m[1] > 35]
        
        if not valid_matches:
            return recipient_query, "no_match"
            
        matched_contacts = []
        top_score = valid_matches[0][1]
        for match in valid_matches:
            if match[1] >= top_score - 20:
                for c in SUPPORTED_CONTACTS:
                    if c["name"].lower() == match[0].lower() and c not in matched_contacts:
                        matched_contacts.append(c)
                        
        if len(matched_contacts) > 1:
            return matched_contacts, "ambiguous"
            
        return matched_contacts[0], "exact"

    def parse(self, transcript: str) -> dict:
        """
        Master execution pipeline for Layer 4. 
        Returns a strict dictionary schema ready for backend processing.
        """
        intent = self.classify_intent(transcript)
        amount = self.extract_amount(transcript)
        
        result_slot = {
            "raw_transcript": transcript,
            "intent": intent,
            "amount": amount,
            "resolved_entity": None,
            "resolution_status": "pending",
            "utility_service": None
        }

        if intent == "transfer_funds":
            entity, status = self.extract_recipient_and_resolve(transcript, amount)
            result_slot["resolved_entity"] = entity
            result_slot["resolution_status"] = status
            
        elif intent == "pay_utility":
            result_slot["utility_service"] = self.extract_utility_provider(transcript)
            result_slot["resolution_status"] = "exact" if amount else "missing_amount"

        return result_slot


# --- TEST HARNESS FOR LAYER 4 ---
if __name__ == "__main__":
    nlu = NPSNluEngine()
    
    # Test cases to verify Layer 4 parser robustness
    test_transcripts = [
        "Send 5000 rupees to Ram Sharma",
        "Ram laai paanch hajar pathaideu",
        "Pay electricity bill of 1500 for NEA",
        "Pay 800 rupees for Khanepani water bill",
        "Send 2000 to Ram" # Tests ambiguity
    ]
    
    print("\n" + "=" * 60)
    print("      RUNNING LAYER 4 NLU PARSER TEST SUITE      ")
    print("=" * 60)
    
    for idx, text in enumerate(test_transcripts, 1):
        print(f"\n[Test {idx}] Input: \"{text}\"")
        structured_json = nlu.parse(text)
        print(f"Output JSON:\n{structured_json}")
    print("\n" + "=" * 60)

    