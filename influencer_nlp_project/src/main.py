"""
Main execution script for BERT and RoBERTa training pipeline
"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from utils import load_and_preprocess_data, prepare_multilabel_data, prepare_sentiment_data
from models import (BERTMultiLabelClassifier, RoBERTaMultiLabelClassifier,
                   BERTSentimentClassifier, RoBERTaSentimentClassifier)
from datasets import MultiLabelDataset, SentimentDataset
from training import train_multilabel_model, train_sentiment_model, evaluate_multilabel_model, evaluate_sentiment_model
from evaluation import calculate_multilabel_metrics, calculate_sentiment_metrics, print_metrics
from inference import save_model_and_tokenizer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    """Main execution function"""
    # Load data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('../data/label_data_Tuan.xlsx')
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ========================================================================
    # MULTI-LABEL CLASSIFICATION
    # ========================================================================
    print("\n" + "="*80)
    print("MULTI-LABEL CLASSIFICATION PIPELINE")
    print("="*80)
    
    # Prepare multi-label data
    X_multi, y_multi, label_columns = prepare_multilabel_data(df)
    
    # Split data (80% train, 20% test)
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi[:, 0]
    )
    
    print(f"Multi-label split: Train={len(X_train_multi)}, Test={len(X_test_multi)}")
    
    # Initialize tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # BERT Multi-Label Training
    print("\n--- Training BERT Multi-Label Classifier ---")
    train_dataset_bert_multi = MultiLabelDataset(X_train_multi, y_train_multi, bert_tokenizer)
    test_dataset_bert_multi = MultiLabelDataset(X_test_multi, y_test_multi, bert_tokenizer)
    
    train_loader_bert_multi = DataLoader(train_dataset_bert_multi, batch_size=16, shuffle=True)
    test_loader_bert_multi = DataLoader(test_dataset_bert_multi, batch_size=16, shuffle=False)
    
    bert_multi_model = BERTMultiLabelClassifier(n_classes=len(label_columns)).to(device)
    train_multilabel_model(bert_multi_model, train_loader_bert_multi, None, num_epochs=3)
    
    # Evaluate BERT multi-label model
    bert_multi_pred, bert_multi_true = evaluate_multilabel_model(bert_multi_model, test_loader_bert_multi)
    bert_multi_metrics = calculate_multilabel_metrics(bert_multi_true, bert_multi_pred, label_columns)
    print_metrics(bert_multi_metrics, "Multi-Label Classification", "BERT")
    
    # RoBERTa Multi-Label Training
    print("\n--- Training RoBERTa Multi-Label Classifier ---")
    train_dataset_roberta_multi = MultiLabelDataset(X_train_multi, y_train_multi, roberta_tokenizer)
    test_dataset_roberta_multi = MultiLabelDataset(X_test_multi, y_test_multi, roberta_tokenizer)
    
    train_loader_roberta_multi = DataLoader(train_dataset_roberta_multi, batch_size=16, shuffle=True)
    test_loader_roberta_multi = DataLoader(test_dataset_roberta_multi, batch_size=16, shuffle=False)
    
    roberta_multi_model = RoBERTaMultiLabelClassifier(n_classes=len(label_columns)).to(device)
    train_multilabel_model(roberta_multi_model, train_loader_roberta_multi, None, num_epochs=3)
    
    # Evaluate RoBERTa multi-label model
    roberta_multi_pred, roberta_multi_true = evaluate_multilabel_model(roberta_multi_model, test_loader_roberta_multi)
    roberta_multi_metrics = calculate_multilabel_metrics(roberta_multi_true, roberta_multi_pred, label_columns)
    print_metrics(roberta_multi_metrics, "Multi-Label Classification", "RoBERTa")
    
    # ========================================================================
    # SENTIMENT ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("="*80)
    
    # Prepare sentiment data
    X_sent, y_sent = prepare_sentiment_data(df)
    
    # Split data (80% train, 20% test)
    X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
        X_sent, y_sent, test_size=0.2, random_state=42, stratify=y_sent
    )
    
    print(f"Sentiment split: Train={len(X_train_sent)}, Test={len(X_test_sent)}")
    
    # BERT Sentiment Training
    print("\n--- Training BERT Sentiment Classifier ---")
    train_dataset_bert_sent = SentimentDataset(X_train_sent, y_train_sent, bert_tokenizer)
    test_dataset_bert_sent = SentimentDataset(X_test_sent, y_test_sent, bert_tokenizer)
    
    train_loader_bert_sent = DataLoader(train_dataset_bert_sent, batch_size=8, shuffle=True)
    test_loader_bert_sent = DataLoader(test_dataset_bert_sent, batch_size=8, shuffle=False)
    
    bert_sent_model = BERTSentimentClassifier(n_classes=3).to(device)
    train_sentiment_model(bert_sent_model, train_loader_bert_sent, None, num_epochs=3)
    
    # Evaluate BERT sentiment model
    bert_sent_pred, bert_sent_true = evaluate_sentiment_model(bert_sent_model, test_loader_bert_sent)
    bert_sent_metrics = calculate_sentiment_metrics(bert_sent_true, bert_sent_pred)
    print_metrics(bert_sent_metrics, "Sentiment Analysis", "BERT")
    
    # RoBERTa Sentiment Training
    print("\n--- Training RoBERTa Sentiment Classifier ---")
    train_dataset_roberta_sent = SentimentDataset(X_train_sent, y_train_sent, roberta_tokenizer)
    test_dataset_roberta_sent = SentimentDataset(X_test_sent, y_test_sent, roberta_tokenizer)
    
    train_loader_roberta_sent = DataLoader(train_dataset_roberta_sent, batch_size=16, shuffle=True)
    test_loader_roberta_sent = DataLoader(test_dataset_roberta_sent, batch_size=16, shuffle=False)
    
    roberta_sent_model = RoBERTaSentimentClassifier(n_classes=3).to(device)
    train_sentiment_model(roberta_sent_model, train_loader_roberta_sent, None, num_epochs=3)
    
    # Evaluate RoBERTa sentiment model
    roberta_sent_pred, roberta_sent_true = evaluate_sentiment_model(roberta_sent_model, test_loader_roberta_sent)
    roberta_sent_metrics = calculate_sentiment_metrics(roberta_sent_true, roberta_sent_pred)
    print_metrics(roberta_sent_metrics, "Sentiment Analysis", "RoBERTa")
    
    # ========================================================================
    # MODEL COMPARISON AND FINAL RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("MODEL COMPARISON AND FINAL RESULTS")
    print("="*80)
    
    # Compare multi-label models
    print("\n🏆 MULTI-LABEL CLASSIFICATION WINNER:")
    bert_multi_f1 = bert_multi_metrics['f1_macro']
    roberta_multi_f1 = roberta_multi_metrics['f1_macro']
    
    if bert_multi_f1 > roberta_multi_f1:
        print(f"BERT wins with Macro F1: {bert_multi_f1:.4f} vs RoBERTa: {roberta_multi_f1:.4f}")
        winner_multi = "BERT"
        best_multi_model = bert_multi_model
        best_multi_tokenizer = bert_tokenizer
    else:
        print(f"RoBERTa wins with Macro F1: {roberta_multi_f1:.4f} vs BERT: {bert_multi_f1:.4f}")
        winner_multi = "RoBERTa"
        best_multi_model = roberta_multi_model
        best_multi_tokenizer = roberta_tokenizer
    
    # Compare sentiment models
    print("\n🏆 SENTIMENT ANALYSIS WINNER:")
    bert_sent_f1 = bert_sent_metrics['f1_macro']
    roberta_sent_f1 = roberta_sent_metrics['f1_macro']
    
    if bert_sent_f1 > roberta_sent_f1:
        print(f"BERT wins with Macro F1: {bert_sent_f1:.4f} vs RoBERTa: {roberta_sent_f1:.4f}")
        winner_sent = "BERT"
        best_sent_model = bert_sent_model
        best_sent_tokenizer = bert_tokenizer
    else:
        print(f"RoBERTa wins with Macro F1: {roberta_sent_f1:.4f} vs BERT: {bert_sent_f1:.4f}")
        winner_sent = "RoBERTa"
        best_sent_model = roberta_sent_model
        best_sent_tokenizer = roberta_tokenizer
    
    # Save best models
    save_model_and_tokenizer(best_multi_model, best_multi_tokenizer, winner_multi, "multilabel")
    save_model_and_tokenizer(best_sent_model, best_sent_tokenizer, winner_sent, "sentiment")
    
    # Summary table
    print("\n📊 COMPREHENSIVE RESULTS SUMMARY:")
    print("-" * 80)
    print(f"{'Task':<25} {'Model':<10} {'Accuracy':<10} {'F1-Macro':<10} {'F1-Micro':<10}")
    print("-" * 80)
    print(f"{'Multi-Label':<25} {'BERT':<10} {bert_multi_metrics['accuracy']:<10.4f} {bert_multi_metrics['f1_macro']:<10.4f} {bert_multi_metrics['f1_micro']:<10.4f}")
    print(f"{'Multi-Label':<25} {'RoBERTa':<10} {roberta_multi_metrics['accuracy']:<10.4f} {roberta_multi_metrics['f1_macro']:<10.4f} {roberta_multi_metrics['f1_micro']:<10.4f}")
    print(f"{'Sentiment':<25} {'BERT':<10} {bert_sent_metrics['accuracy']:<10.4f} {bert_sent_metrics['f1_macro']:<10.4f} {'-':<10}")
    print(f"{'Sentiment':<25} {'RoBERTa':<10} {roberta_sent_metrics['accuracy']:<10.4f} {roberta_sent_metrics['f1_macro']:<10.4f} {'-':<10}")
    print("-" * 80)
    
    # Performance insights
    print("\n💡 KEY INSIGHTS:")
    print(f"• Multi-label classification winner: {winner_multi}")
    print(f"• Sentiment analysis winner: {winner_sent}")
    print(f"• Dataset contains {len(df)} total samples")
    print(f"• Sentiment distribution: {1797/len(df)*100:.1f}% positive, {700/len(df)*100:.1f}% neutral, {271/len(df)*100:.1f}% negative")
    print(f"• Multi-label imbalance: price_value is the most challenging label")
    
    # Deployment recommendations
    print("\n🚀 DEPLOYMENT RECOMMENDATIONS:")
    print("For real-time streaming applications:")
    print(f"• Use {winner_multi} for multi-label classification")
    print(f"• Use {winner_sent} for sentiment analysis")
    print("• Consider model compression (distillation) for production")
    print("• Implement batch processing for better throughput")
    print("• Monitor model drift with incoming streaming data")

if __name__ == "__main__":
    main()