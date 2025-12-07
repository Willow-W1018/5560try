from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

ANSWER_PREFIX = "That is a great question. "
ANSWER_SUFFIX = " Let me know if you have any other questions."
MODEL_PATH = "models/gpt2-squad-formatted"


def load_qa_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
    )
    return generator
