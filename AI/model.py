from utils import load_model_and_tokenizer, generate_answer 
class Assistant:

    def __init__(self, model_name: str = "lora_gpt2_small"):
        self.model, self.tokenizer = load_model_and_tokenizer(model_name=model_name, mode="eval")

    def answer_question(self, question: str, context: str) -> str:
        if not self.model or not self.tokenizer:
            return "Model is unavailable. Please check the logs for details."
        return generate_answer(
            model=self.model,
            tokenizer=self.tokenizer,
            question=question,
            context=context
        )