from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
    """
    Load the specified Mistral model and tokenizer

    Args:
        model_name: Hugging Face model name to load

    Returns:
        tuple of (model, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def get_forced_response_logits(
    messages: List[Dict[str, str]],
    model=None,
    tokenizer=None,
    model_name: str = "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
) -> Tuple[List[str], List[int], np.ndarray]:
    """
    Get logits for a forced assistant response in a chat

    Args:
        messages: List of chat messages in OpenAI format
                 (last one should be the assistant's response)
        model: The loaded language model (loaded if None)
        tokenizer: The model's tokenizer (loaded if None)
        model_name: Name of the model to load if model and tokenizer are None

    Returns:
        tuple of (tokens, token_ids, logits_array) for the assistant's response
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_name)

    # Ensure the last message is from the assistant
    if messages[-1]["role"] != "assistant":
        raise ValueError("Last message must be from the assistant")

    # Use the tokenizer's built-in chat template to format messages
    full_messages = messages.copy()
    messages_without_response = messages[:-1]

    # Apply the chat template to both message sets
    full_prompt = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    prompt_without_response = tokenizer.apply_chat_template(
        messages_without_response, tokenize=False, add_generation_prompt=True
    )

    # Tokenize both versions
    full_inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    inputs_without_response = tokenizer(
        prompt_without_response, return_tensors="pt"
    ).to(model.device)

    # The response starts after the prompt without response
    response_start_idx = inputs_without_response["input_ids"].shape[1]

    # Run forward pass through the model for the full sequence
    with torch.no_grad():
        outputs = model(**full_inputs)

    # Get logits for the assistant's response
    # The logits at position i predict the token at position i+1
    # So we need to offset by -1
    logits = outputs.logits[0, (response_start_idx - 1) : -1]

    # Get the token IDs and tokens for the assistant's response
    response_token_ids = full_inputs["input_ids"][0, response_start_idx:].tolist()
    response_tokens = [tokenizer.decode([id]) for id in response_token_ids]

    # Convert to numpy for easier handling
    logits_array = logits.cpu().numpy()

    return response_tokens, response_token_ids, logits_array


if __name__ == "__main__":
    # Example usage
    model, tokenizer = load_model()

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant called Kona."},
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "My name in reality is Kona."},
    ]

    tokens, token_ids, logits = get_forced_response_logits(messages, model, tokenizer)

    # Print sample results
    print(f"Number of tokens in response: {len(tokens)}")
    print(f"First few tokens: {tokens[:10]}")
    print(f"Logits shape: {logits.shape}")

    # For the first few tokens, show the top predicted tokens
    for i in range(min(5, len(tokens))):
        token_logits = logits[i]
        # Get full probability distribution across entire vocabulary
        full_probs = F.softmax(torch.tensor(token_logits), dim=0)
        # Get the actual probability for the token that was used
        actual_prob = full_probs[token_ids[i]].item()

        # Find top-5 tokens
        top_indices = np.argsort(token_logits)[-5:][::-1]
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
        # Get the probabilities from the FULL distribution (not renormalizing)
        top_probs = [full_probs[idx].item() for idx in top_indices]

        print(
            f"\nActual token: '{tokens[i]}' (ID: {token_ids[i]}, prob: {actual_prob:.4f})"
        )
        print(f"Top 5 predictions:")
        for j, (tok, prob) in enumerate(zip(top_tokens, top_probs)):
            print(f"  {j+1}. '{tok}' with probability {prob:.4f}")
