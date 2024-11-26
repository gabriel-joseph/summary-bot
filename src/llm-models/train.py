from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

model_name = "sshleifer/distilbart-cnn-12-6"
save_dir = "./distilbart_model"

# Download and save the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model and tokenizer saved to {save_dir}")
