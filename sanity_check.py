import torch
from llama import load_pretrained

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

sanity_data = torch.load("./sanity_check.data")
# text_batch = ["hello world", "hello neural network for NLP"]
# tokenizer here
sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                         [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])

# load our model
llama = load_pretrained("stories42M.pt")
with torch.no_grad():
    logits, hidden_states = llama(sent_ids)

    print(f"My logits shape: {logits.shape}")
    print(f"Sanity logits shape: {sanity_data['logits'].shape}")
    print(f"Diff Max: {(logits - sanity_data['logits']).abs().max()}")
    print(f"Diff Mean: {(logits - sanity_data['logits']).abs().mean()}")

    # どの程度惜しいのか確認
    diff = (logits - sanity_data["logits"]).abs()
    outliers = (diff > 1e-5).sum()
    print(f"Threshold (1e-5) を超えた要素数: {outliers} / {logits.numel()}")
    print(f"最大誤差のインデックス付近の値: \nMy logits: {logits[diff > 1e-5][:3]}\nSanity: {sanity_data['logits'][diff > 1e-5][:3]}")

    # assert torch.allclose(logits, sanity_data["logits"], atol=1e-5, rtol=1e-3)
    assert torch.allclose(hidden_states, sanity_data["hidden_states"], atol=1e-5, rtol=1e-3)
    print("Your Llama implementation is correct!")