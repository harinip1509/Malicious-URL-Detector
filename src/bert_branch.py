"""The semantic branch: a fine-tuned DistilBERT that reads the URL as text.

It produces two things the rest of the system needs. The malicious probability
becomes one of the five votes, and the CLS embedding becomes 768 extra columns
for the boosting models, so the trees get semantic context they could never
derive from character counts alone.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import config
from .features import to_sentence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UrlDataset(Dataset):
    def __init__(self, urls, labels, tokenizer, max_len):
        self.enc = tokenizer([to_sentence(u) for u in urls], truncation=True,
                             padding="max_length", max_length=max_len,
                             return_tensors="pt")
        self.y = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


def fine_tune(train_df, val_df, cfg, out_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.BERT_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_NAME, num_labels=2).to(DEVICE)

    dl_tr = DataLoader(UrlDataset(train_df["url"], train_df["y"], tok, cfg.max_len),
                       batch_size=cfg.batch_size, shuffle=True)
    dl_va = DataLoader(UrlDataset(val_df["url"], val_df["y"], tok, cfg.max_len),
                       batch_size=cfg.batch_size * 2)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(dl_tr), pct_start=0.1)

    for ep in range(cfg.epochs):
        model.train()
        running = 0.0
        for i, batch in enumerate(dl_tr):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += out.loss.item()
            if i % 50 == 0:
                print(f"  epoch {ep+1} step {i}/{len(dl_tr)} loss {running/(i+1):.4f}",
                      end="\r")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in dl_va:
                y = batch.pop("labels").to(DEVICE)
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                correct += (model(**batch).logits.argmax(-1) == y).sum().item()
                total += len(y)
        print(f"\n  epoch {ep+1}: validation accuracy {correct/total:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return model, tok


def load(out_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(out_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(out_dir)).to(DEVICE)
    model.eval()
    return model, tok


@torch.no_grad()
def infer(model, tokenizer, urls, max_len=config.MAX_LEN, batch_size=256):
    """Return (malicious probability, CLS embedding) for a list of URLs."""
    model.eval()
    texts = [to_sentence(u) for u in urls]
    probs, embs = [], []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True,
                        padding="max_length", max_length=max_len,
                        return_tensors="pt").to(DEVICE)
        out = model(**enc, output_hidden_states=True)
        probs.append(torch.softmax(out.logits, -1)[:, 1].cpu().numpy())
        embs.append(out.hidden_states[-1][:, 0, :].cpu().numpy())
    return np.concatenate(probs), np.vstack(embs)
