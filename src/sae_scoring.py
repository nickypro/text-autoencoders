# %% Full Script for Automated SAE Feature Interpretation (Simplified with utils_gen)

import os
import json
import torch
from tqdm import tqdm # Use standard tqdm
import numpy as np
import random
import time
import gc # Garbage collector
import re # For regex parsing

# --- Environment Variable Loading ---
try:
    from dotenv import load_dotenv
    # Specify path if your .env file is not in the root directory
    # load_dotenv(dotenv_path="path/to/.env")
    if load_dotenv(): # Use default .env lookup
         print("Loaded environment variables from .env file.")
    else:
         print("No .env file found or python-dotenv not installed, relying on system environment variables.")
except ImportError:
    print("python-dotenv not installed, relying on system environment variables.")

# --- Project-Specific Imports & Definitions ---

# Use the provided utility for parallel API calls
try:
    from utils_gen import get_prompts_parallel
    USE_UTILS_GEN = True
    print("Successfully imported 'get_prompts_parallel' from utils_gen.")
except ImportError:
    print("ERROR: Could not import 'get_prompts_parallel' from utils_gen.py.")
    print("       Ensure utils_gen.py and utils_parallel.py are accessible.")
    USE_UTILS_GEN = False

# Assuming utils_load_data.py is in the same directory or accessible
from utils_load_data import load_embeds, load_split_paragraphs

# Assuming utils_sonar.py with the tokenizer loader is accessible
try:
    from sonar.models.sonar_text import load_sonar_tokenizer
    # Updated load_tokenizer function provided by user
    def load_tokenizer(repo="text_sonar_basic_encoder"):
        # print(f"Loading SONAR tokenizer from repo: {repo}") # Reduce noise
        orig_tokenizer = load_sonar_tokenizer(repo)
        tokenizer = orig_tokenizer.create_encoder()
        vocab_size = orig_tokenizer.vocab_info.size

        class TokenizerWrapper:
            def __init__(self, tokenizer, vocab_size):
                self.tokenizer = tokenizer
                self.vocab_size = vocab_size
                self._orig_tokenizer = orig_tokenizer # Store for decoder access

            def __getattr__(self, name):
                return getattr(self.tokenizer, name)
            def __call__(self, *args, **kwargs):
                return self.tokenizer(*args, **kwargs)
            def encode(self, text):
                return self.tokenizer(text)
            def encode_as_tokens(self, text):
                 if hasattr(self.tokenizer, 'encode_as_tokens'):
                     return self.tokenizer.encode_as_tokens(text)
                 else:
                     # print("Warning: encode_as_tokens not available on the wrapped encoder.")
                     return None
            def get_decoder(self):
                 if hasattr(self._orig_tokenizer, 'create_decoder'):
                     return self._orig_tokenizer.create_decoder()
                 else:
                     return None

        wrapped_tokenizer = TokenizerWrapper(tokenizer, vocab_size)
        # print(f"Tokenizer loaded. Vocab size: {wrapped_tokenizer.vocab_size}")
        return wrapped_tokenizer

except ImportError:
    print("ERROR: SONAR library not found or load_sonar_tokenizer failed.")
    class DummyTokenizer:
        vocab_size = 10
        def __init__(self): self._vocab_size=10
        def encode(self, text): return torch.tensor([random.randint(0,9) for _ in text.split()])
        def get_decoder(self): return None
    tokenizer = DummyTokenizer()
except Exception as e:
    print(f"An unexpected error occurred during tokenizer loading: {e}")
    raise

# --- SAE Model Definition (Must match the trained model) ---
class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, l1_coeff): # l1_coeff not needed for inference
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=True)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        encoded = self.encoder(x)
        activations = self.relu(encoded)
        reconstructed = self.decoder(activations)
        return reconstructed, activations
    def encode(self, x):
        encoded = self.encoder(x)
        activations = self.relu(encoded)
        return activations
    def decode(self, activations):
        return self.decoder(activations)
# --- End SAE Model Definition ---


# --- Configuration ---
# Data & Model Paths
SAE_MODEL_DIR = "./sae_models"
# *** IMPORTANT: Specify the exact model and config file you want to interpret ***
SAE_MODEL_FILENAME = "sae_final_H8192_L1_8.0e-06.pt" # Example: Use the better model
SAE_CONFIG_FILENAME = "sae_config_H8192_L1_8.0e-06.json"
SAE_MODEL_PATH = os.path.join(SAE_MODEL_DIR, SAE_MODEL_FILENAME)
SAE_CONFIG_PATH = os.path.join(SAE_MODEL_DIR, SAE_CONFIG_FILENAME)

# Interpretation Parameters
NUM_SAMPLES_FOR_MADE = 100000 # How many samples to calculate activations over to find MADE
MAX_FILES_TO_TRY_MADE = 200  # Max files to load data from for MADE calculation
NUM_FEATURES_TO_INTERPRET = 20 # How many features to analyze (e.g., first N, set None for all)
NUM_MADE_EXAMPLES = 20       # Number of top activating examples per feature
NUM_NEG_EXAMPLES = 20        # Number of bottom activating (negative) examples per feature
MAX_CHARS_PER_EXAMPLE = 600  # Truncate long examples (adjust based on typical text length)
INTERPRETATION_BATCH_SIZE = 512 # Batch size for calculating activations

# API Call Parameters (using utils_gen)
MAX_WORKERS = 10 # Number of parallel threads for API calls (passed to utils_gen)

# Output Configuration
OUTPUT_DIR = "./sae_interpretations"
OUTPUT_FILENAME = f"feature_desc_{SAE_MODEL_FILENAME.replace('.pt', '')}.json"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# System & Reproducibility
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

# --- Set Seed ---
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# --- Helper Functions ---

def load_sae_model(model_path, config_path, device):
    """Loads a trained SAE model."""
    print(f"Loading SAE config from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loading SAE model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = SparseAutoencoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        l1_coeff=0
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Warning: Failed loading state_dict with weights_only=True ({e}). Trying with weights_only=False.")
        print("         Ensure you trust the model source file.")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    print("SAE model loaded successfully.")
    return model, config

def load_interpretation_data(num_samples, max_files_to_try=50):
    """Loads embeddings and texts for finding MADE examples."""
    all_embeddings = []
    all_texts = []
    files_loaded = 0
    print(f"Attempting to load {num_samples} samples for interpretation...")

    # Use standard tqdm here
    pbar = tqdm(total=num_samples, desc="Loading interpretation data")
    initial_count = 0
    for i in range(max_files_to_try):
        if len(all_embeddings) >= num_samples:
            pbar.update(num_samples - initial_count) # Ensure progress bar reaches 100%
            break
        try:
            embeds_batch = load_embeds(i)
            texts_batch = load_split_paragraphs(i)
            files_loaded += 1
            loaded_in_file = 0
            for embed, text in zip(embeds_batch, texts_batch):
                if len(text) > 0:
                    all_embeddings.append(embed.cpu())
                    all_texts.append(text)
                    loaded_in_file += 1
                    if len(all_embeddings) >= num_samples:
                        break
            pbar.update(loaded_in_file)
            initial_count = len(all_embeddings)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: Error loading file {i}: {e}")
            continue
    pbar.close()

    if len(all_embeddings) < num_samples:
        print(f"Warning: Could only load {len(all_embeddings)} samples out of {num_samples} requested.")
    else:
        print(f"Successfully loaded {len(all_embeddings)} samples from {files_loaded} files.")

    return all_embeddings, all_texts

@torch.no_grad()
def get_feature_activations(sae_model, embeddings_tensor, batch_size, device):
    """Get activations for all features across the dataset."""
    sae_model.eval()
    all_activations = []
    print("Calculating SAE activations for the dataset...")
    # Use standard tqdm
    for i in tqdm(range(0, embeddings_tensor.shape[0], batch_size), desc="Getting Activations"):
        batch = embeddings_tensor[i:i+batch_size].to(device)
        activations = sae_model.encode(batch)
        all_activations.append(activations.cpu())
        del batch, activations
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
    return torch.cat(all_activations, dim=0)

def get_activating_examples(feature_idx, activations, texts, k_pos=NUM_MADE_EXAMPLES, k_neg=NUM_NEG_EXAMPLES):
    """Find texts that maximally and minimally activate a given feature."""
    if feature_idx >= activations.shape[1]:
        print(f"Warning: feature_idx {feature_idx} out of bounds.")
        return [], []
    feature_activation_values = activations[:, feature_idx]

    # Get top k positive examples
    top_indices = torch.argsort(feature_activation_values, descending=True)[:k_pos]
    top_examples = []
    for idx in top_indices:
        text = texts[idx.item()]
        if len(text) > MAX_CHARS_PER_EXAMPLE:
             text = text[:MAX_CHARS_PER_EXAMPLE] + "..."
        top_examples.append({
            "text": text,
            "activation": feature_activation_values[idx].item()
        })

    # Get bottom k negative examples
    bottom_indices = torch.argsort(feature_activation_values, descending=False)[:k_neg]
    bottom_examples = []
    for idx in bottom_indices:
        text = texts[idx.item()]
        if len(text) > MAX_CHARS_PER_EXAMPLE:
             text = text[:MAX_CHARS_PER_EXAMPLE] + "..."
        bottom_examples.append({
            "text": text,
            "activation": feature_activation_values[idx].item() # Still store activation, even if low/zero
        })

    return top_examples, bottom_examples

def format_prompt(feature_idx, top_examples, bottom_examples):
    """Creates the prompt for the OpenAI API call, including positive and negative examples."""
    prompt_header = f"""You are an AI assistant specialized in interpreting features from neural networks trained on text data.

I have a feature (neuron #{feature_idx}) from a Sparse Autoencoder's hidden layer.

The following {len(top_examples)} text snippets cause the *highest* activation value for this feature. A higher activation value means the feature is more 'active' or relevant for that specific text input.

--- High Activating Examples ---
"""
    top_examples_str = ""
    for i, example in enumerate(top_examples):
        safe_text = example['text'].replace('"', "'").replace('\\', '/')
        top_examples_str += f"\n{i+1}. Activation: {example['activation']:.4f}\n   Text: \"{safe_text}\"\n"

    negative_header = f"""
--- End High Activating Examples ---

Additionally, the following {len(bottom_examples)} text snippets are randomly sampled to have caused *low* activation values for this feature. These 'negative examples' show text where the feature is *not* strongly present or relevant, which can help clarify what the feature *is not* about by providing contrast.

--- Low Activating Examples (Negative Examples) ---
"""
    bottom_examples_str = ""
    for i, example in enumerate(bottom_examples):
        safe_text = example['text'].replace('"', "'").replace('\\', '/')
        # Negative examples might have 0 activation, format appropriately
        bottom_examples_str += f"\n{i+1}. Activation: {example['activation']:.4f}\n   Text: \"{safe_text}\"\n"

    prompt_footer = f"""
--- End Low Activating Examples ---

Analyze *both* the high and low activating examples. Identify the common pattern, concept, topic, or linguistic feature in the high activating examples that seems *absent* or *contrasted* in the low activating examples. What specific, recurring element seems to be triggering the feature's activation, considering this contrast?

Based ONLY on the provided examples, please provide your analysis strictly as a JSON object matching the following schema:

```json
{{
  "reasoning": "string | Your brief step-by-step analysis considering both positive and negative examples and the justification for the label. Mention specific examples if helpful.",
  "label": "string | A concise descriptive label (2-5 words) for this feature.",
  "explanation": "string | A brief explanation (1-2 sentences) of the pattern identified, ideally highlighting the contrast.",
  "specificity": "number | Estimate the feature's scope (1=Broad/Abstract, 5=Very Specific):\n    5: Individual Entity/Item (e.g., specific person, product model, unique place).\n    4: Narrow Category/Class (e.g., programming languages, dog breeds, capital cities).\n    3: Broad Category/Topic (e.g., science, sports, food, emotions, common actions).\n    2: General Abstract Concept / Grammatical Function (e.g., sentiment, questions, tense, possibility).\n 1: very broad type of text, e.g 'english speech'.",
  "polysemy_detected": "boolean | True if you suspect the feature responds to multiple distinct concepts based on these examples, False otherwise. A heuristic for this is if you feel like you need to use 'and' to describe the feature.",
  "confidence": "number | Your confidence in this interpretation on a scale of 1 (low) to 5 (high), based *only* on the provided examples."
}}
```

Output ONLY the JSON object. Do not include any text before or after the JSON object. If no clear pattern emerges even with the contrast, indicate this in the reasoning/explanation and assign low confidence (e.g., 1).
"""
    return prompt_header + top_examples_str + negative_header + bottom_examples_str + prompt_footer

def parse_llm_json_response(response_text, feature_idx):
    """Attempts to parse JSON from the LLM response, handling potential errors."""
    # Handle empty response case from utils_gen error handling
    if not response_text:
         return {"label": "API Error", "explanation": "Empty response from API call.", "raw_response": "", "confidence": 0, "polysemy_detected": False, "reasoning": ""}
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1) or match.group(2)
            parsed_data = json.loads(json_str)
            interpretation = {
                "reasoning": parsed_data.get("reasoning", "N/A"),
                "label": parsed_data.get("label", "Parsing Error"),
                "explanation": parsed_data.get("explanation", "N/A"),
                "confidence": parsed_data.get("confidence", 0),
                "specificity": parsed_data.get("specificity", 0),
                "polysemy_detected": parsed_data.get("polysemy_detected", False),
                "raw_response": response_text
            }
            if not isinstance(interpretation["confidence"], (int, float)):
                try: interpretation["confidence"] = float(interpretation["confidence"])
                except: interpretation["confidence"] = 0
            if not isinstance(interpretation["polysemy_detected"], bool):
                 interpretation["polysemy_detected"] = str(interpretation["polysemy_detected"]).lower() == 'true'
            return interpretation
        else:
             print(f"Warning: No JSON block found in response for feature {feature_idx}.")
             return {"label": "Parsing Error", "explanation": "No JSON block found", "raw_response": response_text, "confidence": 0, "polysemy_detected": False, "reasoning": ""}
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error for feature {feature_idx}: {e}\nResponse Text: {response_text[:500]}...")
        return {"label": "JSON Decode Error", "explanation": str(e), "raw_response": response_text, "confidence": 0, "polysemy_detected": False, "reasoning": ""}
    except Exception as e:
        print(f"Other Parsing Error for feature {feature_idx}: {e}\nResponse Text: {response_text[:500]}...")
        return {"label": "Parsing Error", "explanation": str(e), "raw_response": response_text, "confidence": 0, "polysemy_detected": False, "reasoning": ""}


# --- Main Interpretation Logic ---
def main():
    if not USE_UTILS_GEN:
        print("Exiting: utils_gen.py could not be imported.")
        return

    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load SAE Model and Config
    sae_model, sae_config = load_sae_model(SAE_MODEL_PATH, SAE_CONFIG_PATH, DEVICE)
    HIDDEN_DIM = sae_config['hidden_dim']

    # 2. Load Tokenizer (Optional)
    # tokenizer = load_tokenizer()

    # 3. Load Data (Embeddings and Texts) for finding MADE examples
    embeddings_list, texts = load_interpretation_data(NUM_SAMPLES_FOR_MADE, MAX_FILES_TO_TRY_MADE)
    if not embeddings_list:
        print("No data loaded for interpretation. Exiting.")
        return
    embeddings_tensor = torch.stack(embeddings_list)
    print(f"Data loaded: {embeddings_tensor.shape[0]} samples.")

    # 4. Get Activations for the entire dataset
    activations = get_feature_activations(sae_model, embeddings_tensor, INTERPRETATION_BATCH_SIZE, DEVICE)
    del embeddings_tensor # Free memory
    gc.collect()
    print(f"Activations calculated: Shape {activations.shape}")

    # 5. Prepare Prompts for Features to Interpret
    num_features = min(NUM_FEATURES_TO_INTERPRET or HIDDEN_DIM, HIDDEN_DIM)
    feature_indices_to_process = list(range(num_features))
    prompts_with_indices = [] # Store tuples of (feature_idx, prompt)

    print(f"\n--- Preparing Prompts for {num_features} Features ---")
    for feature_idx in tqdm(feature_indices_to_process, desc="Generating Prompts"):
        top_examples, bottom_examples = get_activating_examples(
            feature_idx,
            activations,
            texts,
            k_pos=NUM_MADE_EXAMPLES,
            k_neg=NUM_NEG_EXAMPLES
        )
        if not top_examples: # If no positive examples, likely an issue or dead feature
            print(f"Warning: No top examples found for feature {feature_idx}. Skipping.")
            continue
        # It's okay if bottom_examples is empty, though less ideal for contrast
        prompt = format_prompt(feature_idx, top_examples, bottom_examples)
        prompts_with_indices.append((feature_idx, prompt))

    if not prompts_with_indices:
        print("No prompts generated. Exiting.")
        return

    # 6. Call OpenAI API in Parallel using utils_gen
    print(f"\n--- Calling LLM for {len(prompts_with_indices)} Features (Max Workers: {MAX_WORKERS}) ---")
    prompts_only = [p for idx, p in prompts_with_indices]
    start_time = time.time()
    # Assuming get_prompts_parallel shows its own progress bar
    completions = get_prompts_parallel(prompts_only, max_workers=MAX_WORKERS)
    end_time = time.time()
    print(f"LLM calls completed in {end_time - start_time:.2f} seconds.")

    # 7. Process and Parse Results
    print("\n--- Processing and Parsing LLM Responses ---")
    all_interpretations = {}
    if len(completions) != len(prompts_with_indices):
        print(f"Error: Mismatch between number of prompts ({len(prompts_with_indices)}) and completions ({len(completions)}).")
        # Attempt partial processing if possible
        min_len = min(len(completions), len(prompts_with_indices))
    else:
        min_len = len(completions)

    for i in tqdm(range(min_len), desc="Parsing Results"):
        feature_idx = prompts_with_indices[i][0]
        completion_text = completions[i]
        parsed_result = parse_llm_json_response(completion_text, feature_idx)
        # Add top/bottom examples to the saved results for reference
        # Find the original examples again (or store them alongside prompts_with_indices earlier)
        top_ex, bottom_ex = get_activating_examples(feature_idx, activations, texts, k_pos=NUM_MADE_EXAMPLES, k_neg=NUM_NEG_EXAMPLES)
        parsed_result['top_examples'] = top_ex
        parsed_result['bottom_examples'] = bottom_ex
        # Store the prompt used for this feature as well
        parsed_result['prompt_used'] = prompts_with_indices[i][1]
        all_interpretations[feature_idx] = parsed_result
        # Remove prompt from list to avoid duplication if needed, although storing it per feature is better
        # parsed_result['prompt'] = prompts_only # Old line - removed

    # 8. Save Results
    print(f"\nSaving {len(all_interpretations)} interpretations to {OUTPUT_FILE}...")
    all_interpretations_serializable = {str(k): v for k, v in all_interpretations.items()}
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_interpretations_serializable, f, indent=4)
        print("Interpretations saved successfully.")
    except Exception as e:
        print(f"Error saving interpretations to JSON: {e}")

    # 9. Display Sample Results
    print("\n--- Sample Interpretations ---")
    display_count = 0
    # Sort by feature index for display
    for idx in sorted(all_interpretations.keys()):
        if display_count >= 15: break
        interp = all_interpretations[idx]
        # Remove redundant prompt display here as it's now part of the interp dict
        # if 'prompt' in interp and interp['prompt']:
        #    print("  Prompt:")
        #    print(interp['prompt'])

        print(f"\nFeature {idx}:")
        print(f"  Label: {interp.get('label', 'N/A')}")
        print(f"  Explanation: {interp.get('explanation', 'N/A')}")
        print(f"  Specificity: {interp.get('specificity', 'N/A')}")
        print(f"  Confidence: {interp.get('confidence', 'N/A')}")
        print(f"  Polysemy Detected: {interp.get('polysemy_detected', 'N/A')}")
        print(f"  Reasoning: {interp.get('reasoning', 'N/A')}")
        # Display top/bottom examples if available
        if interp.get('top_examples'):
             print(f"  Top Example 1: {interp['top_examples'][0]['text'][:100]}... (Act: {interp['top_examples'][0]['activation']:.3f})")
        if interp.get('bottom_examples'):
             print(f"  Bottom Example 1: {interp['bottom_examples'][0]['text'][:100]}... (Act: {interp['bottom_examples'][0]['activation']:.3f})")
        # print(f"  Reasoning: {interp.get('reasoning', 'N/A')}") # Optional
        display_count += 1

# --- Main Execution Block ---
if __name__ == "__main__":
    main()
    print("\nInterpretation script finished.")

# %%