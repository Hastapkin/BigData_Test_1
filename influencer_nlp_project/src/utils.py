"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    df = pd.read_excel(file_path, sheet_name='Labeled_Data')
    
    # Drop rows with missing comments
    df = df.dropna(subset=['comment'])
    
    # Clean text
    df['comment'] = df['comment'].astype(str).str.strip()
    
    # Remove empty comments
    df = df[df['comment'] != '']
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    return df

def prepare_multilabel_data(df):
    """Prepare data for multi-label classification"""
    # Select multi-label columns
    label_columns = ['brand', 'influencer', 'product_quality', 'price_value', 
                    'purchase_intent', 'experience_feedback']
    
    X = df['comment'].values
    y = df[label_columns].values.astype(int)
    
    print(f"Multi-label data prepared:")
    print(f"Input shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution:")
    for i, col in enumerate(label_columns):
        positive_count = np.sum(y[:, i])
        print(f"  {col}: {positive_count}/{len(y)} ({positive_count/len(y)*100:.1f}%)")
    
    return X, y, label_columns

def prepare_sentiment_data(df):
    """Prepare data for sentiment analysis"""
    X = df['comment'].values
    y = df['sentiment_no'].values.astype(int)
    
    print(f"Sentiment data prepared:")
    print(f"Input shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        sentiment_name = ['negative', 'neutral', 'positive'][label]
        print(f"  {label} ({sentiment_name}): {count}/{len(y)} ({count/len(y)*100:.1f}%)")
    
    return X, y