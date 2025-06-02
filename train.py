import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from llama import LLaMA  # 방금 구현한 LLaMA 클래스를 불러옵니다.


# --- 커스텀 Dataset 정의 ---
class TextDataset(Dataset):
    def __init__(self, input_ids, block_size):
        self.input_ids = input_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.input_ids) - self.block_size

    def __getitem__(self, idx):
        x = self.input_ids[idx : idx + self.block_size]
        y = self.input_ids[idx + 1 : idx + 1 + self.block_size]
        return x, y


# --- 학습 함수 ---
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(dataloader, desc="Training"):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        # logits shape: (batch_size, seq_len, vocab_size)

        # CrossEntropyLoss expects (N, C) pred and (N,) target
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 토크나이저 로드 (예시로 GPT2 사용)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # WikiText-2 로드
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = raw_datasets["train"]["text"]
    texts = [t for t in texts if t.strip()]
    texts = texts[:100]  # 빠른 예시를 위해 일부만 사용

    # 전체 텍스트 연결 후 토크나이즈
    full_text = " ".join(texts)
    encodings = tokenizer(full_text, return_tensors="pt", padding=False)
    input_ids = encodings["input_ids"].squeeze(0)  # (total_seq_len,)

    # Dataset & DataLoader
    block_size = 128
    dataset = TextDataset(input_ids, block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )

    # LLaMA 모델 생성
    dim = 256
    n_layers = 4
    heads = 8
    hidden_dim = 4 * dim
    max_len = block_size

    model = LLaMA(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        n_layers=n_layers,
        heads=heads,
        hidden_dim=hidden_dim,
        max_len=max_len,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    # 학습 루프
    epochs = 3
    for epoch in range(1, epochs + 1):
        avg_loss = train(model, dataloader, optimizer, device)
        print(f"[Epoch {epoch:02d}] Average Loss: {avg_loss:.4f}")
        torch.save(model, "model.pth")


if __name__ == "__main__":
    main()
