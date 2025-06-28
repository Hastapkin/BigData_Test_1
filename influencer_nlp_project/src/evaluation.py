"""
Evaluation metrics for multi-label and sentiment classification
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           confusion_matrix, hamming_loss)

def calculate_multilabel_metrics(y_true, y_pred, label_names):
    """Calculate comprehensive metrics for multi-label classification"""
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)
    
    # Micro and macro metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Per-label metrics
    precision_per_label, recall_per_label, f1_per_label, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'hamming_loss': hamming,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }
    
    # Add per-label metrics
    for i, label in enumerate(label_names):
        metrics[f'{label}_precision'] = precision_per_label[i]
        metrics[f'{label}_recall'] = recall_per_label[i]
        metrics[f'{label}_f1'] = f1_per_label[i]
    
    return metrics

def calculate_sentiment_metrics(y_true, y_pred):
    """Calculate comprehensive metrics for sentiment classification"""
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro and weighted metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm
    }
    
    # Add per-class metrics
    class_names = ['negative', 'neutral', 'positive']
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = precision_per_class[i]
        metrics[f'{class_name}_recall'] = recall_per_class[i]
        metrics[f'{class_name}_f1'] = f1_per_class[i]
    
    return metrics

def print_metrics(metrics, task_name, model_name):
    """Print metrics in a formatted way"""
    print(f"\n{'='*60}")
    print(f"{task_name} - {model_name} Results")
    print(f"{'='*60}")
    
    if 'accuracy' in metrics:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    if 'hamming_loss' in metrics:
        print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"Micro F1: {metrics['f1_micro']:.4f}")
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
        print(f"Micro Precision: {metrics['precision_micro']:.4f}")
        print(f"Micro Recall: {metrics['recall_micro']:.4f}")
        print(f"Macro Precision: {metrics['precision_macro']:.4f}")
        print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    else:
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        print(f"Precision Macro: {metrics['precision_macro']:.4f}")
        print(f"Recall Macro: {metrics['recall_macro']:.4f}")
        print(f"Precision Weighted: {metrics['precision_weighted']:.4f}")
        print(f"Recall Weighted: {metrics['recall_weighted']:.4f}")
        
        if 'confusion_matrix' in metrics:
            print(f"\nConfusion Matrix:")
            print(metrics['confusion_matrix'])