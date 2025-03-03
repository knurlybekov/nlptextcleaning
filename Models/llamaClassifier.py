from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
huggingFaceToken = "" #needs token from hugging face, ask Ethan
trainingData = "./sampledDataTrain.csv" #Path to training data
validationData = "./sampledDataTest.csv" #Path to validation data
#Make sure the label in training and validation csv is headed as "category" and text is labled as "text_cl"

df = pd.read_csv(trainingData)
label_encoder = LabelEncoder()
df["category"] = label_encoder.fit_transform(df["category"])
encoding_map = {i: label for i, label in enumerate(label_encoder.classes_)}
print(encoding_map)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token = huggingFaceToken) 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(repr(tokenizer.pad_token))

class TextClassificationDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["fixed_text"]
        label = self.df.iloc[idx]["category"]

        encoding = self.tokenizer(
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
            "category": torch.tensor(label, dtype=torch.long),
        }
    
dataset = TextClassificationDataset(df, tokenizer)

loader = DataLoader(dataset, batch_size=8, shuffle=True)
print(f"Batches: {len(loader)}")

model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B", num_labels = 19, token = huggingFaceToken)
model.config.pad_token_id = model.config.eos_token_id

device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
torch.cuda.empty_cache()
try:
    for epoch in range(1):
        print("starting epoch: " + str(epoch))
        model.train()
        total_loss = 0
        batchNum = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["category"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batchNum += 1
            print(f"Epoch {epoch+1}, Batch: {batchNum}, Loss: {total_loss / len(loader)}")
except RuntimeError as e:
    print("RuntimeError:", e)


model.eval()

testData = pd.read_csv(validationData)
testData["category"] = label_encoder.transform(testData["category"])

testdataset = TextClassificationDataset(testData, tokenizer)
testLoader = DataLoader(testdataset, batch_size=8, shuffle=False)


correct = 0
total = 0
tBatch = 0
predicted_labels = []
actual_labels = []

with torch.no_grad():
    for batch in testLoader:
        print(f"Starting test batch {tBatch}")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["category"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        if outputs.logits is None:
            print("Error: No logits found in model output.")
            continue
        _, predicted = torch.max(outputs.logits, dim=1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        tBatch += 1

        try:
            predicted_labels.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
        except Exception:
            print("That didn't work")

accuracy = correct / len(testLoader.dataset)

print(f"Accuracy: {accuracy}")
report = classification_report(actual_labels, predicted_labels, output_dict=True)
print("Classification Report:")
for label, metrics in report.items():
    print(f"Label: {label}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1-score']:.4f}")
    print(f"Support: {metrics['support']}")
    print()


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
