from transformers import LlamaForSequenceClassification, LlamaTokenizer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import pandas as pd

# Load pre-trained model and tokenizer
model = LlamaForSequenceClassification.from_pretrained("llama-base")
tokenizer = LlamaTokenizer.from_pretrained("llama-base")
trainingData = pd.read_csv("pathToTrainingData")
testData = pd.read_csv("pathToTestData")


# Set up optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-5)
loss_fn = CrossEntropyLoss()

# Train the model
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

    model.eval()
eval_loss = 0
correct = 0
with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs, labels)
        eval_loss += loss.item()
        _, predicted = torch.max(outputs.scores, dim=1)
        correct += (predicted == labels).sum().item()

accuracy = correct / len(val_dataloader.dataset)
print(f"Validation Loss: {eval_loss / len(val_dataloader)}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Save the best-performing model
torch.save(model.state_dict(), "best_model.pth")