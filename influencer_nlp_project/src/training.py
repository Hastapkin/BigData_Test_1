"""
Training functions for BERT and RoBERTa models
"""
import torch
import torch.nn as nn
from torch.optim import AdamW  # Fixed import
from transformers import get_linear_schedule_with_warmup
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Rest of your training.py code remains the same...

def train_multilabel_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
    """Train multi-label classification model"""
    try:
        criterion = nn.BCELoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Print progress every 10 batches
                    if batch_idx % 10 == 0:
                        print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                        
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
            
            # Validation
            if val_loader:
                try:
                    val_loss = evaluate_multilabel_model(model, val_loader, criterion)
                    print(f'Validation Loss: {val_loss:.4f}')
                except Exception as e:
                    print(f"Validation error: {e}")
                    
    except Exception as e:
        print(f"Training error: {e}")
        raise e

def train_sentiment_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
    """Train sentiment classification model"""
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    # FIX: Convert labels to tensor properly
                    labels = batch['labels']
                    if isinstance(labels, list):
                        labels = torch.tensor(labels, dtype=torch.long).to(device)
                    else:
                        labels = labels.to(device)
                    
                    # Ensure labels are 1D
                    if labels.dim() > 1:
                        labels = labels.squeeze()
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Print progress every 10 batches
                    if batch_idx % 10 == 0:
                        print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                        
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
                    
    except Exception as e:
        print(f"Training error: {e}")
        raise e

def evaluate_multilabel_model(model, data_loader, criterion=None):
    """Evaluate multi-label classification model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            predictions = (outputs > 0.5).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    if criterion:
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    else:
        return all_predictions, all_labels

def evaluate_sentiment_model(model, data_loader, criterion=None):
    """Evaluate sentiment classification model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # FIX: Handle labels properly
            labels = batch['labels']
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long).to(device)
            else:
                labels = labels.to(device)
            
            if labels.dim() > 1:
                labels = labels.squeeze()
            
            outputs = model(input_ids, attention_mask)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    if criterion:
        avg_loss = total_loss / len(data_loader)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(all_labels, all_predictions)
        return avg_loss, accuracy
    else:
        return all_predictions, all_labels