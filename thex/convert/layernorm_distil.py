# layernorm distillation from bert

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertConfig
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class LinearNormBERT(BertModel):
    def __init__(self, config):
        super(LinearNormBERT, self).__init__(config)
        for layer in self.encoder.layer:
            layer.output.LayerNorm = nn.Linear(config.hidden_size, config.hidden_size)

def compute_metrics(pred, labels):
    pred_labels = np.argmax(pred, axis=1)
    accuracy = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels, average="macro")
    return {"accuracy": accuracy, "f1": f1}

def evaluate(model, val_dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in val_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return compute_metrics(all_logits, all_labels)


class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits):
        student_probs = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        loss = self.kl_div_loss(student_probs, teacher_probs)
        return loss

def distillation_train_step(model, inputs, teacher_model, distillation_loss, optimizer):
    model.train()
    optimizer.zero_grad()

    student_outputs = model(**inputs)
    student_logits = student_outputs.logits

    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs)
        teacher_logits = teacher_outputs.logits

    loss = distillation_loss(student_logits, teacher_logits)
    loss.backward()
    optimizer.step()

    return loss.item()

def copy_weights(pretrained_model, linear_norm_bert_model):
    model_dict = linear_norm_bert_model.state_dict()
    pretrained_dict = pretrained_model.state_dict()
    new_dict = {}

    for k, v in pretrained_dict.items():
        if k in model_dict:
            if "LayerNorm" not in k:
                new_dict[k] = v

    model_dict.update(new_dict)
    linear_norm_bert_model.load_state_dict(model_dict)

    return linear_norm_bert_model

def collate_fn(batch):
    keys = batch[0].keys()
    batch_tensors = {}
    
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            batch_tensors[key] = torch.stack([example[key] for example in batch])
        else:
            batch_tensors[key] = [example[key] for example in batch]
    
    return batch_tensors

def run_distill():

    pretrained_bert = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
    config = BertConfig.from_pretrained(pretrained_bert)
    pretrained_model = BertModel.from_pretrained(pretrained_bert)
    
    bert_model = LinearNormBERT(config)
    bert_model = copy_weights(pretrained_model, bert_model)

    teacher_model = BertForSequenceClassification.from_pretrained(pretrained_bert, return_dict=True)
    teacher_model.eval()

    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=128)

    dataset = load_dataset("glue", "mrpc")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    teacher_model.to(device)

    num_epochs = 3
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    optimizer = torch.optim.Adam(bert_model.parameters(), lr=1e-5)
    distillation_loss = DistillationLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch in train_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            loss = distillation_train_step(bert_model, inputs, teacher_model, distillation_loss, optimizer)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

        eval_metrics = evaluate(bert_model, val_dataloader, device)
        print(f"Validation - Accuracy: {eval_metrics['accuracy']:.4f}, F1: {eval_metrics['f1']:.4f}")



if __name__ == "__main__":
    run_distill()
