import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import torch 

def load_model_and_tokenizer(model_name=None, mode="train", BASE_MODEL_NAME="gpt2", DEVICE=None):

    if DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Case 1: None or invalid path -> create empty LoRA GPT-2
    if model_name is None or not os.path.exists(model_name):
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
    print(f"Loading LoRA model from: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    base_model.config.use_cache = False

    # is_trainable=True ensures LoRA weights are attached for training
    is_trainable = mode == "train"
    model = PeftModel.from_pretrained(base_model, model_name, is_trainable=is_trainable)

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
    max_new_tokens: int = 128,
    temperature: float = 0.3,
):
    """
    Generate answer for a given question + context using GPT-2 LoRA model
    """

    model.eval()

    prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
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

    # Optional: stop if model continues into next section
    for stop_token in ["\nContext:", "\nQuestion:"]:
        if stop_token in answer:
            answer = answer.split(stop_token)[0].strip()

    return answer
# loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_name="lora_gpt2_small", mode="eval")