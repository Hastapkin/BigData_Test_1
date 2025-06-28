import torch
from transformers import BertTokenizer, BertModel

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Test simple BERT loading
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    # Test inference
    text = "This is a test"
    inputs = tokenizer(text, return_tensors='pt').to(device)
    outputs = model(**inputs)
    
    print("✅ BERT model loaded and working!")
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")