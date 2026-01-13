import os
import numpy as np

# ✅ 改这里：把 main 换成你实际存放这些函数的 py 文件名
from main import read_one_jsonl, _sentence_split, _clean_text, tokenize_for_sklearn, summarize_textrank_mmr

from sklearn.feature_extraction.text import TfidfVectorizer

def summary_A_lead(content: str, k: int = 3, max_chars: int = 220):
    """A：前k句截断"""
    sents = _sentence_split(content)
    if not sents:
        return _clean_text(content)[:max_chars]
    return "。".join(sents[:k])[:max_chars]

def summary_B_textrank_plain(content: str, max_sents: int = 3, max_chars: int = 220):
    """B：传统句子级 TextRank（三句），不加信息密度/不做MMR"""
    content = _clean_text(content)
    if not content:
        return ""
    sents = _sentence_split(content)
    if not sents:
        return content[:max_chars]
    if len(sents) <= max_sents:
        return "。".join(sents)[:max_chars]

    sents = sents[:30]
    n = len(sents)

    vec = TfidfVectorizer(tokenizer=tokenize_for_sklearn, token_pattern=None)
    X = vec.fit_transform(sents)

    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0.0)

    row_sum = sim.sum(axis=1, keepdims=True) + 1e-12
    P = sim / row_sum

    d = 0.85
    pr = np.ones(n) / n
    for _ in range(40):
        pr = (1 - d) / n + d * (P.T @ pr)

    # 传统TextRank：直接取PR TopK
    idx = np.argsort(-pr)[:max_sents]
    idx = sorted(idx.tolist())  # 按原文顺序拼接
    out = "。".join(sents[i] for i in idx).strip()
    return out[:max_chars]

def pick_one_file(data_dir="examples"):
    for name in os.listdir(data_dir):
        if name.endswith(".jsonl") and name != "result_log.jsonl":
            return os.path.join(data_dir, name)
    return None

if __name__ == "__main__":
    path = pick_one_file("examples")
    if not path:
        raise FileNotFoundError("examples/ 下没找到可用的 .jsonl 文件")

    title, content = read_one_jsonl(path)

    A = summary_A_lead(content, k=3, max_chars=220)
    B = summary_B_textrank_plain(content, max_sents=3, max_chars=220)
    C = summarize_textrank_mmr(content, max_chars=220, max_sents=3)

    print("=== R5：同一文档三种摘要对比 ===")
    print("文件：", os.path.basename(path))
    print("Title：", title)
    print("\n[A] 前3句截断：")
    print(A)
    print("\n[B] 传统 TextRank（三句）：")
    print(B)
    print("\n[C] 改进 TextRank（信息密度+MMR 三句）：")
    print(C)
