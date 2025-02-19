from transformers import LlamaForSequenceClassification, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
from huggingface_hub import Repository

df = pd.read_csv("./small_df.csv")
label_encoder = LabelEncoder()
df["category"] = label_encoder.fit_transform(df["category"])

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

class TextClassificationDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text_cl"]
        label = self.df.iloc[idx]["category"]

        encoding = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            #pad_token=self.tokenizer.eos_token_id,
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "category": torch.tensor(label, dtype=torch.long),
        }
    
dataset = TextClassificationDataset(df, tokenizer)

loader = DataLoader(dataset, batch_size=8, shuffle=True)
print(len(loader))

model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.1-8B", num_labels = 19)
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
try:
    for epoch in range(5):
        print("starting epoch: " + str(epoch))
        model.train()
        total_loss = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["category"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader)}")
except RuntimeError as e:
    print("RuntimeError:", e)


model.eval()

testLoader = DataLoader(dataset, batch_size=32, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for batch in testLoader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["category"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        if outputs.logits is None:
            print("Error: No logits found in model output.")
            continue
        _, predicted = torch.max(outputs.logits, dim=1)

        correct += (predicted == labels).sum().item()

accuracy = correct / len(testLoader.dataset)

print(accuracy)


save_dir = "./trainedModels"
os.makedirs(save_dir, exist_ok=True)

# Move model to CPU before saving
model.cpu()

try:
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Model and tokenizer saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

# Move model back to GPU if needed
model.to(device)
