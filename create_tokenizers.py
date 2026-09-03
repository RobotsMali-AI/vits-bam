import os
import json
import re
from datasets import load_dataset
from tqdm import tqdm

# ----------------------------------------------------
# 1. TEXT CLEANING & NORMALIZATION SCHEMES
# ----------------------------------------------------

def clean_bambara_pseudo_ipa(text):
    """
    Translates pure Bambara text into pseudo-IPA form.
    Normalizes nasalizations and specific consonants (c -> tʃ, j -> dʒ).
    """
    if not text or not isinstance(text, str):
        return ""
        
    text = text.lower()
    
    # Target replacement dictionary for Bambara Nasal Vowels
    nasal_map = {
        "aan": "ãã", "ɛɛn": "ɛ̃ɛ̃", "een": "ẽẽ", "iin": "ĩĩ", "ɔɔn": "ɔ̃ɔ̃", "oon": "õõ", "uun": "ũũ",
        "an": "ã", "ɛn": "ɛ̃", "en": "ẽ", "in": "ĩ", "ɔn": "ɔ̃", "on": "õ", "un": "ũ"
    }

    # Step A: Apply Nasal Vowel mappings
    for ortho, ipa in nasal_map.items():
        text = re.sub(rf"{ortho}(?![aeɛioɔu])", ipa, text)
    
    # Step B: Align unique sound representations
    sound_alignments = {
        "j": "dʒ",     
        "c": "tʃ",     
        "ɲ": "ɲ",     
        "ŋ": "ŋ"      
    }
    for b_char, ipa_char in sound_alignments.items():
        text = text.replace(b_char, ipa_char)
        
    # Step C: Final cleanup to drop weird hidden unicode artifacts 
    # (Keeping standard alphabet, bambara specials, tildes, and our new ʃ/ʒ)
    text = re.sub(r"[^a-zɛɔɲŋãẽĩõṹ̀̂̌̄\s.,!?ʃʒ]", "", text)
    
    # Clean up excess spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# ----------------------------------------------------
# 2. DATASET PARSING & TOKENIZER GENERATION
# ----------------------------------------------------

def generate_tokenizer_from_dataset(dataset_id, config_name, text_column, split, save_dir):
    print(f"Loading dataset: {dataset_id} ({config_name}) - Split: {split}")
    # Load dataset (streams text to save RAM/Time if it's huge, but mapping is fast enough)
    dataset = load_dataset(dataset_id, config_name, split=split)
    
    unique_chars = set()
    
    print("Normalizing text and extracting unique characters...")
    for item in tqdm(dataset):
        raw_text = item.get(text_column, "")
        cleaned_text = clean_bambara_pseudo_ipa(raw_text)
        unique_chars.update(list(cleaned_text))
        
    # Sort for deterministic output
    vocab_list = sorted(list(unique_chars))
    print(f"\nExtracted {len(vocab_list)} unique characters.")
    print("Characters:", "".join(vocab_list))
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct vocabulary dictionary with VITS structural parameters
    vocab = {"_": 0, "<unk>": 1, " ": 2, "c": 3, "j": 4} # Keep c and j in the vocab for flexibility. 
    
    for char in vocab_list:
        if char not in vocab:
            vocab[char] = len(vocab)
            
    # Include default VITS sentence ending markers just in case they were missing from the dataset
    for marker in [".", ",", "!", "?", " "]:
        if marker not in vocab:
            vocab[marker] = len(vocab)

    # 1. Output vocab.json
    with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 2. Output tokenizer_config.json
    tokenizer_config = {
        "add_blank": True,
        "added_tokens_decoder": {
            "0": {
                "content": "_",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "clean_up_tokenization_spaces": True,
        "is_uroman": False,
        "language": None,
        "model_max_length": 1000000000000000019884624838656,
        "normalize": False, 
        "pad_token": "_",
        "phonemize": False,
        "tokenizer_class": "VitsTokenizer",
        "unk_token": "<unk>",
        "verbose": False
    }
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        
    # 3. Output special_tokens_map.json
    special_tokens_map = {
        "pad_token": "_",
        "unk_token": "<unk>"
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)
        
    # 4. Output added_tokens.json
    added_tokens = {
        "<unk>": 1
    }
    with open(os.path.join(save_dir, "added_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(added_tokens, f, ensure_ascii=False, indent=2)

    print(f"Tokenizer files successfully saved to: {save_dir}")

if __name__ == "__main__":
    generate_tokenizer_from_dataset(
        dataset_id="Panga-Azazia/all-in-one",
        config_name="afvoices",
        text_column="text",
        split="train",
        save_dir="./vits_bam_tokenizer"
    )