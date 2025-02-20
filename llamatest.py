from transformers import LlamaForSequenceClassification, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
from huggingface_hub import Repository

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("./trainedModels", num_labels = 19)
idMap = ['arts', 'crime', 'disaster', 'economy', 'education', 'environmental', 'health',
 'human interest', 'humaninterest', 'labour', 'lifestyle', 'other', 'politics',
 'religion', 'science', 'social', 'sport', 'unrest', 'weather']
# Create a sample input
sample_text = "police officer assault arrest police seen marked outside police department headquarters press carry dock police officer assault charge prosecution service statement saying charge cost reminder sigh service charge arrest person downtown service allegation status police department would say remains active member first schedule court appearance provincial court government fight court action lost appealing decision"

# Preprocess the input using the tokenizer
inputs = tokenizer(sample_text, return_tensors="pt")

# Move the inputs to the device (GPU or CPU)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Make predictions using the model
outputs = model(**inputs)

# Get the predicted class label
predicted_label = torch.argmax(outputs.logits)

# Print the predicted label
print(f"Predicted label: {idMap[predicted_label]}")