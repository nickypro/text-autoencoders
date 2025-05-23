# %% FUNCTION THAT GETS CE LOSS FOR SONAR DECODER
from sonar.models.sonar_text import (
    load_sonar_text_decoder_model,
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)
import torch
import torch.nn.functional as F
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.data.text import TextTokenDecoder
from sonar.nn.conditional_decoder_model import ConditionalTransformerDecoderModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer(repo="text_sonar_basic_encoder"):
    return load_sonar_tokenizer(repo).create_encoder()

class SonarDecoderCELoss(torch.nn.Module):
    def __init__(self,
            repo="text_sonar_basic_decoder",
            lang="eng_Latn",
            device=DEVICE,
            dtype=None,
            preloaded_decoder=None,
            preloaded_tokenizer=None,
            skip_first_token=True,
            max_tokens=None
        ):
        super().__init__()
        if preloaded_decoder is None:
            # self.model: ConditionalTransformerDecoderModel = vec2text.model
            vec2text = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=device)
            self.vec2text = vec2text
            self.model = vec2text.model
            self.decoder = self.model.decoder
            self.tokenizer = vec2text.tokenizer

        else:
            self.model: ConditionalTransformerDecoderModel = preloaded_decoder
            self.decoder = self.model.decoder
            self.tokenizer = preloaded_tokenizer

        self.model.requires_grad_(True)
        self.tokenizer_decoder = self.tokenizer.create_decoder()
        self.tokenizer_encoder = self.tokenizer.create_encoder(device=device)
        self.device = device
        self.lang = lang
        self.skip_first_token = skip_first_token
        self.max_tokens = max_tokens if max_tokens is not None else 512
        print(type(self.model))

    def tokenize_text(self, expected_text):    # Tokenize the expected text
        if isinstance(expected_text, str):
            expected_text = [expected_text]

        expected_tokens = [self.tokenizer_encoder(text) for text in expected_text]
        expected_tokens = [tokens[:self.max_tokens] for tokens in expected_tokens]

        # IMPORTANT: Add BOS token to the start of the expected tokens
        bos_tensor = torch.tensor([3], device=expected_tokens[0].device, dtype=expected_tokens[0].dtype)
        expected_tokens = [torch.cat([bos_tensor, tokens]) for tokens in expected_tokens]

        # Create padding mask and collate the expected tokens
        max_len = max(len(tokens) for tokens in expected_tokens)
        padded_tokens = []
        padding_mask_list = []

        for tokens in expected_tokens:
            pad_len = max_len - len(tokens)
            padded = torch.cat([tokens, torch.full((pad_len,), self.tokenizer.vocab_info.pad_idx,
                                                device=tokens.device, dtype=tokens.dtype)])
            padded_tokens.append(padded)
            padding_mask_list.append(torch.cat([torch.ones(len(tokens), dtype=torch.bool, device=tokens.device),
                                            torch.zeros(pad_len, dtype=torch.bool, device=tokens.device)]))

        target_tokens = torch.stack(padded_tokens).to(self.device)
        stacked_padding_mask = torch.stack(padding_mask_list)

        # Create PaddingMask with both the mask and sequence length
        seq_lens = torch.sum(stacked_padding_mask, dim=1).to(torch.int)
        padding_mask = PaddingMask(stacked_padding_mask, seq_lens)

        # Split into input tokens (all but last) and target tokens (all but first)
        input_tokens = target_tokens[:, :-1]  # Remove last token
        target_labels = target_tokens[:, 1:]  # Remove first token (BOS)

        # Create appropriate padding masks
        input_seq_lens = torch.clamp(seq_lens - 1, min=0)
        input_padding_mask = PaddingMask(input_seq_lens, batch_seq_len=max_len - 1)

        return {
            "original_tokens": expected_text,
            "input_tokens": input_tokens,
            "target_labels": target_labels,
            "input_padding_mask": input_padding_mask,
            "padding_mask": padding_mask
        }

    def get_logits(self, input_embed, input_tokens, input_padding_mask, **kwargs):
        # Prepare encoder output (our embedding)
        # Ensure input_embed is properly shaped [batch_size, 1, embedding_dim]
        if input_embed.dim() == 2:
            input_embed = input_embed.unsqueeze(1)

        encoder_output = input_embed
        encoder_padding_mask = None

        # Get decoder frontend features (token embeddings + positional encoding)
        seqs, padding_mask_dec = self.decoder.decoder_frontend(
            input_tokens, input_padding_mask)

        # Run through decoder
        decoder_output, _ = self.decoder.decoder(
            seqs.to(self.device),
            padding_mask=padding_mask_dec.to(self.device),
            encoder_output=encoder_output.to(self.device),
            encoder_padding_mask=encoder_padding_mask
        )

        # Get logits by projecting
        logit_data = self.decoder.project(decoder_output, padding_mask_dec)
        return logit_data.logits

    def _skip_first_token(self, target_labels):
        # Create copy to avoid inplace modification
        modified_labels = target_labels.clone()
        for b in range(modified_labels.size(0)):
            for t in range(modified_labels.size(1)):
                if modified_labels[b, t] != self.tokenizer.vocab_info.pad_idx:
                    modified_labels[b, t] = self.tokenizer.vocab_info.pad_idx
                    break
        return modified_labels

    def calculate_loss(self, logits, target_labels):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_labels.reshape(-1),
            ignore_index=self.tokenizer.vocab_info.pad_idx
        )


    def calculate_accuracy(self, logits, token_data, verbose=False):
        probs = F.softmax(logits, dim=-1)

        # Track per-token probabilities and predictions
        batch_size, seq_len, vocab_size = logits.shape
        token_probs = []
        correct_count = 0
        total_count = 0

        print("\nTOKEN PREDICTIONS:")
        vocab = self.tokenizer.vocab_info

        for b in range(batch_size):
            print("-"*12)
            for s in range(seq_len):
                # Skip padding tokens
                if s >= token_data["input_padding_mask"].seq_lens[b]:
                    continue

                # Get the predicted and target token
                pred_idx = torch.argmax(logits[b, s]).item()
                target_idx = token_data["target_labels"][b, s].item()

                # Skip padding in target
                if target_idx == vocab.pad_idx:
                    continue

                # Track statistics
                total_count += 1
                correct = (pred_idx == target_idx)
                if correct:
                    correct_count += 1

                # Get probability of the target token
                target_prob = probs[b, s, target_idx].item()
                token_probs.append(target_prob)

                # Print token information
                prefix = "✓" if correct else "✗"
                dec = lambda x: self.tokenizer_decoder(torch.tensor([x]))
                if verbose:
                    print(f"{prefix} Position {s+1}: Target prob={target_prob:.4f}, Predicted: {pred_idx} '{dec(pred_idx)}', Expected: {target_idx} '{dec(target_idx)}'")

        accuracy_data = {
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "avg_prob": sum(token_probs) / len(token_probs) if token_probs else 0,
        }

        return accuracy_data

    def get_ce_loss_decoder(self, input_embed, expected_text):
        token_data = self.tokenize_text(expected_text)
        logits = self.get_logits(input_embed, **token_data)

        with torch.no_grad():
            token_data = self.tokenize_text(expected_text)
            if self.skip_first_token:
                token_data["target_labels"] = self._skip_first_token(token_data["target_labels"])

        loss = self.calculate_loss(logits, token_data["target_labels"])

        return loss

    def __call__(self, input_embed, expected_text):
        return self.get_ce_loss_decoder(input_embed, expected_text)

    def verbose_loss(self, input_embed, expected_text, verbose=True):
        token_data = self.tokenize_text(expected_text)
        logits = self.get_logits(input_embed, **token_data)

        if self.skip_first_token:
            token_data["target_labels"] = self._skip_first_token(token_data["target_labels"])

        # Print summary statistics
        loss = self.calculate_loss(logits, token_data["target_labels"])
        accuracy_data = self.calculate_accuracy(logits, token_data, verbose=verbose)

        print(f"\nSUMMARY:")
        print(f"Accuracy: {accuracy_data['accuracy']:.2f}")
        print(f"Average target token probability: {accuracy_data['avg_prob']:.4f}")
        print(f"Loss: {loss.item():.4f}")
        return loss

    def decode(self, embeddings, target_lang="eng_Latn", max_seq_len=512):
        return self.vec2text.predict(embeddings, target_lang=target_lang, max_seq_len=max_seq_len)


if __name__ == "__main__":
    # Load up the models
    text2vec = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)
    vec2text = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)
    model: ConditionalTransformerDecoderModel = vec2text.model
    model.requires_grad_(True)

    # sonar_decoder = SonarDecoderCELoss(preloaded_decoder=model, preloaded_tokenizer=vec2text.tokenizer)
    sonar_decoder = SonarDecoderCELoss()
    example_texts = ["Greetings from the Sonar team!", "Hello, how are you?"]

    # test with random input
    print("Checking LOSS with random input")
    input_embed = torch.randn(2, 1, 1024)
    loss = sonar_decoder.verbose_loss(input_embed, example_texts)
    print()

    # check that original model can reconstruct
    input_embed = text2vec.predict(example_texts, source_lang="eng_Latn")
    output_text = vec2text.predict(input_embed, target_lang="eng_Latn")
    print(output_text)

    # check with good input
    with torch.no_grad():
        print("Checking LOSS with good input")
        input_embed_new = input_embed.clone().requires_grad_(True)
        loss = sonar_decoder.verbose_loss(input_embed_new, example_texts)
        print(loss)

    # check with max_tokens
    with torch.no_grad():
        sonar_decoder.max_tokens = 5
        input_embed_new = input_embed.clone().requires_grad_(True)
        loss = sonar_decoder.verbose_loss(input_embed_new, example_texts)
        print(loss)

    # check with gradients
    input_embed_new.requires_grad_(True)
    loss = sonar_decoder.verbose_loss(input_embed_new, example_texts)
    print(loss)

    # check loss gradients
    loss.backward()
    print(input_embed_new.grad)
# %%
