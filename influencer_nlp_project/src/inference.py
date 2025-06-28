"""
Inference utilities for production deployment
"""
import torch
import os
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model_and_tokenizer(model, tokenizer, model_name, task_type):
    """Save trained model and tokenizer for deployment"""
    save_dir = f"models/{task_type}_{model_name.lower()}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), f"{save_dir}/model.pt")
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    print(f"Model and tokenizer saved to {save_dir}")

def load_model_for_inference(model_class, tokenizer_class, model_path, n_classes):
    """Load trained model for inference"""
    model = model_class(n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = tokenizer_class.from_pretrained(os.path.dirname(model_path))
    
    return model, tokenizer

def predict_single_comment(model, tokenizer, comment, task_type="sentiment"):
    """Make prediction on a single comment"""
    model.eval()
    
    encoding = tokenizer(
        comment,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        
        if task_type == "sentiment":
            prediction = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            confidence = torch.softmax(outputs, dim=1).max().cpu().numpy()
            sentiment_labels = ['negative', 'neutral', 'positive']
            return sentiment_labels[prediction], confidence
        else:  # multi-label
            predictions = (outputs > 0.5).cpu().numpy()[0]
            confidences = outputs.cpu().numpy()[0]
            label_names = ['brand', 'influencer', 'product_quality', 'price_value', 
                          'purchase_intent', 'experience_feedback']
            
            active_labels = [label_names[i] for i, pred in enumerate(predictions) if pred == 1]
            return active_labels, confidences

def batch_predict_comments(model, tokenizer, comments, task_type="sentiment", batch_size=32):
    """Make predictions on a batch of comments"""
    model.eval()
    results = []
    
    for i in range(0, len(comments), batch_size):
        batch_comments = comments[i:i+batch_size]
        
        encodings = tokenizer(
            batch_comments,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            
            if task_type == "sentiment":
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                confidences = torch.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
                
                sentiment_labels = ['negative', 'neutral', 'positive']
                batch_results = [(sentiment_labels[pred], conf) for pred, conf in zip(predictions, confidences)]
            else:  # multi-label
                predictions = (outputs > 0.5).cpu().numpy()
                confidences = outputs.cpu().numpy()
                
                label_names = ['brand', 'influencer', 'product_quality', 'price_value', 
                              'purchase_intent', 'experience_feedback']
                
                batch_results = []
                for pred_row, conf_row in zip(predictions, confidences):
                    active_labels = [label_names[j] for j, pred in enumerate(pred_row) if pred == 1]
                    batch_results.append((active_labels, conf_row))
            
            results.extend(batch_results)
    
    return results