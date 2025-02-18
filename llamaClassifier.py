from transformers import LlamaForSequenceClassification, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv("./small_df.csv")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

class TextClassificationDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        label = self.df.iloc[idx]["label"]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
    
dataset = TextClassificationDataset(df, tokenizer)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", num_labels = 18)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader)}")

model.eval()

testLoader = DataLoader(dataset, batch_size=32, shuffle=False)
correct = 0
with torch.no_grad():
    for batch in testLoader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.scores, dim=1)

        correct += (predicted == labels).sum().item()

accuracy = correct / len(testLoader.dataset)

print(accuracy)

model.save_pretrained("./trainedModels")
tokenizer.save_pretrained("./trainedModels")
