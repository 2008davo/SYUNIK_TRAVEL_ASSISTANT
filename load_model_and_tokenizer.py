import torch

def load_model_and_tokenizer(model_path):
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    # 1. Re-initialize the tokenizer (must be the same as during training)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Re-create the model's architecture (this is an untrained model initially)
    # It must match the architecture that was used when saving the model.
    loaded_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # 3. Define the path to your saved .pkl file
    model_path = 'my_gpt2_qa_model.pkl'
    
    # 4. Load the state_dict (the trained weights) from the .pkl file
    # This overwrites the randomly initialized weights of `loaded_model`
    loaded_model.load_state_dict(torch.load(model_path))
    
    # 5. Move the loaded model to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model #.to(device)
    
    # 6. Set the model to evaluation mode (important for consistent predictions)
    loaded_model.eval()
    
    print(f"Model loaded successfully from {model_path} and moved to {device}")
    print(f"Loaded model architecture and state: {loaded_model}")
    return loaded_model, tokenizer