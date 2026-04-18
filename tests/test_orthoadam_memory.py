import torch
from transformers import GPT2Config, GPT2LMHeadModel
from xai_repro.attention import inject_softmax1
from xai_repro.optim import OrthoAdam

def test_dry_run():
    print("Initializing model...")
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=6,
        n_head=12,
    ) # roughly 60M parameters by halving layers
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel(config)
    model = inject_softmax1(model)
    model.to(device)
    
    print("Initializing OrthoAdam...")
    optimizer = OrthoAdam(model.parameters(), lr=1e-3)
    
    print("Creating dummy batch...")
    batch_size = 2
    seq_len = 256
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    labels = input_ids.clone()
    
    print("Forward pass...")
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    
    print("Backward pass...")
    loss.backward()
    
    print("Optimizer step...")
    optimizer.step()
    
    print("Success!")

if __name__ == '__main__':
    test_dry_run()
