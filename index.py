!pip install transformers sentencepiece --quiet

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate a quote
def generate_quote(prompt):
    input_text = f"Write an inspirational quote about: {prompt}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Try with custom topics
topics = [
    "Confidence",
    "self-love",
    "Kindness"
]

for t in topics:
    print(f"\n📝 Topic: {t}")
    print("💬 Quote:", generate_quote(t))