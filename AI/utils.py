import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import torch 

def load_model_and_tokenizer(model_name=None, mode="train", BASE_MODEL_NAME="gpt2", DEVICE=None):

    if DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Case 1: None or invalid path -> create empty LoRA GPT-2
    base_path = os.path.dirname(__file__)
    
    # Build the full model path, checking nested structure
    model_path = os.path.join(base_path, model_name) if model_name else None
    
    # Check if model exists - handle nested paths like lora_gpt2/lora_gpt2_1/lora_gpt2_1/
    model_exists = False
    if model_path and os.path.exists(model_path):
        # Check if adapter_config.json exists in the directory or nested subdirectories
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            model_exists = True
        else:
            # Check in nested subdirectories (e.g., lora_gpt2/lora_gpt2_1/lora_gpt2_1/)
            for root, dirs, files in os.walk(model_path):
                if "adapter_config.json" in files:
                    model_path = root
                    model_exists = True
                    break
    
    if model_name is None or not model_exists:
        print("No valid model found. Creating empty LoRA GPT-2 model...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        # Load base GPT-2
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
        base_model.config.use_cache = False
        base_model.to(DEVICE)

        # Create empty LoRA
        lora_config = LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["c_attn", "c_proj"],
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        if mode == "train":
            model.train()
        else:
            model.eval()
        return model, tokenizer

    # Case 2: Load existing LoRA adapter
    print(f"Loading LoRA model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    base_model.config.use_cache = False

    # is_trainable=True ensures LoRA weights are attached for training
    is_trainable = mode == "train"
    model = PeftModel.from_pretrained(base_model, model_path, is_trainable=is_trainable)

    model.to(DEVICE)
    if mode == "train":
        model.train()
    else:
        model.eval()

    # Print trainable params
    # model.print_trainable_parameters()

    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    question: str,
    context: str,
    max_new_tokens: int = 100,
    temperature: float = 0.5,
):
    """
    Generate a concise answer for a given question + context using
    the GPT-2 LoRA model. The model is instructed to answer briefly
    and avoid repetition.
    """

    model.eval()

    prompt = (
        "You are a friendly, knowledgeable, and enthusiastic tour guide for the Syunik region of Armenia. "
        "Answer the user's questions using ONLY the provided context. "
        "Keep your answers concise (1-2 sentences), informative, and easy to read. "
        "Add a touch of friendliness or delight, as if speaking to a curious traveler. "
        "Do not repeat yourself or provide unrelated information. "
        # "Only include locations or directions if the question specifically asks for them. "
        "Highlight interesting historical, cultural, or natural facts whenever relevant, "
        "and always remain accurate and helpful.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    # 🔑 Extract only the answer part
    if "Answer:" in decoded:
        answer = decoded.split("Answer:", 1)[-1].strip()
    else:
        answer = decoded.strip()

    # Cut at first blank line to avoid long rambles
    if "\n\n" in answer:
        answer = answer.split("\n\n", 1)[0].strip()

    # Optional: stop if model continues into next section markers
    for stop_token in ["\nContext:", "\nQuestion:"]:
        if stop_token in answer:
            answer = answer.split(stop_token)[0].strip()
    print("Lora GPT-2 answer: ", answer)
    return answer
# loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_name="lora_gpt2_small", mode="eval")