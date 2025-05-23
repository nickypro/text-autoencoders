# %%
import torch
import numpy as np
import plotly.express as px
import pandas as pd
from phate import PHATE
from sklearn.metrics.pairwise import cosine_similarity

# --- SONAR Model Loading ---
try:
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    print("Attempting to load SONAR models...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the Sonar text-to-embedding model
    text2vec = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder", # Although not explicitly used for predict, good practice
        device=device
    )
    print("SONAR text-to-embedding model loaded successfully.")
    SONAR_AVAILABLE = True

except ImportError:
    print("ERROR: SONAR library not found.")
    print("Cannot generate embeddings. Please install SONAR.")
    SONAR_AVAILABLE = False
except Exception as e:
    print(f"An unexpected error occurred during SONAR model loading: {e}")
    SONAR_AVAILABLE = False
    raise # Re-raise the exception after printing the message

# %%
if __name__ == "__main__" and SONAR_AVAILABLE:

    print("\n--- Multi-Sentence Embedding Analysis ---")

    # 1. Define Sentences
    sentence1 = "The cat sat on the mat"
    sentence2 = "The dog chased the ball"

    # 2. Create Combinations
    texts_to_embed = {
        "s1": sentence1,
        "s2": sentence2,
        "s1s2": f"{sentence1}. {sentence2}", # Concatenated S1 then S2
        "s2s1": f"{sentence2}. {sentence1}", # Concatenated S2 then S1
    }
    labels = list(texts_to_embed.keys())
    text_list = list(texts_to_embed.values())

    print("\nTexts to Embed:")
    for label, text in texts_to_embed.items():
        print(f"  {label}: \"{text}\"")

    # 3. Generate Embeddings
    print("\nGenerating embeddings...")
    embeddings = {}
    embeddings_list = []
    try:
        with torch.no_grad():
            # Note: Batch processing is usually more efficient
            # results = text2vec.predict(text_list, "eng_Latn") # returns a list of tensors
            # for i, label in enumerate(labels):
            #     emb = results[i].cpu().numpy()
            #     embeddings[label] = emb
            #     embeddings_list.append(emb)

            # Process one by one for clarity in this example
            for label, text in texts_to_embed.items():
                 embedding = text2vec.predict([text], "eng_Latn")[0] # Get the first (only) embedding
                 emb_np = embedding.cpu().numpy()
                 embeddings[label] = emb_np
                 embeddings_list.append(emb_np)
                 print(f"  Generated embedding for '{label}' (Shape: {emb_np.shape})")

        embeddings_array = np.array(embeddings_list)

        # 4. Compare Embeddings using Cosine Similarity
        print("\nCosine Similarities:")
        # Compare combined sentences to their components
        sim_s1s2_s1 = cosine_similarity(embeddings["s1s2"].reshape(1, -1), embeddings["s1"].reshape(1, -1))[0][0]
        sim_s1s2_s2 = cosine_similarity(embeddings["s1s2"].reshape(1, -1), embeddings["s2"].reshape(1, -1))[0][0]
        sim_s2s1_s1 = cosine_similarity(embeddings["s2s1"].reshape(1, -1), embeddings["s1"].reshape(1, -1))[0][0]
        sim_s2s1_s2 = cosine_similarity(embeddings["s2s1"].reshape(1, -1), embeddings["s2"].reshape(1, -1))[0][0]
        sim_s1s2_s2s1 = cosine_similarity(embeddings["s1s2"].reshape(1, -1), embeddings["s2s1"].reshape(1, -1))[0][0]

        print(f"  sim(s1s2, s1)  : {sim_s1s2_s1:.4f}")
        print(f"  sim(s1s2, s2)  : {sim_s1s2_s2:.4f}")
        print(f"  sim(s2s1, s1)  : {sim_s2s1_s1:.4f}")
        print(f"  sim(s2s1, s2)  : {sim_s2s1_s2:.4f}")
        print(f"  sim(s1s2, s2s1): {sim_s1s2_s2s1:.4f}")

        # Optional: Compare combined embedding to average of components
        avg_s1_s2 = (embeddings["s1"] + embeddings["s2"]) / 2
        sim_s1s2_avg = cosine_similarity(embeddings["s1s2"].reshape(1, -1), avg_s1_s2.reshape(1, -1))[0][0]
        sim_s2s1_avg = cosine_similarity(embeddings["s2s1"].reshape(1, -1), avg_s1_s2.reshape(1, -1))[0][0]
        print(f"  sim(s1s2, avg(s1, s2)): {sim_s1s2_avg:.4f}")
        print(f"  sim(s2s1, avg(s1, s2)): {sim_s2s1_avg:.4f}")


        # 5. Visualize Embeddings using PHATE
        print("\nVisualizing embeddings using PHATE...")
        phate_operator = PHATE(n_components=2, random_state=42, n_jobs=1) # n_jobs=1 for small data
        embeddings_2d = phate_operator.fit_transform(embeddings_array)

        # Create a DataFrame for Plotly
        df = pd.DataFrame({
            'PHATE1': embeddings_2d[:, 0],
            'PHATE2': embeddings_2d[:, 1],
            'label': labels,
            'text': text_list
        })

        # Create interactive plot
        fig = px.scatter(
            df, x='PHATE1', y='PHATE2',
            text='label',
            hover_data=['text'],
            title='PHATE Visualization of Single vs. Combined Sentence Embeddings'
        )

        # Customize appearance
        fig.update_traces(
            textposition='top center',
            marker=dict(size=10)
        )
        fig.update_layout(
            xaxis_title='PHATE 1',
            yaxis_title='PHATE 2',
            width=800,
            height=600
        )

        # Save and show
        output_dir = '../data'
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.write_html(os.path.join(output_dir,'multi_sentence_embeddings_phate.html'))
        # fig.write_image(os.path.join(output_dir,'multi_sentence_embeddings_phate.png')) # Requires kaleido
        fig.show()

        print(f"\nAnalysis complete. Visualization saved to '{os.path.join(output_dir,'multi_sentence_embeddings_phate.html')}'")

    except NameError:
        print("Error: Required variables (e.g., embeddings) not defined. Did embedding generation fail?")
    except Exception as e:
        print(f"An error occurred during analysis or visualization: {e}")

elif not SONAR_AVAILABLE:
    print("Exiting script because SONAR models could not be loaded.")

# %%