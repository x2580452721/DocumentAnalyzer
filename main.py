# -*- coding: utf-8 -*-
import os
import re
import json
import time
import csv
import platform
import subprocess
from collections import Counter

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import jieba
import jieba.analyse
import jieba.posseg as pseg

# ========= 方案A依赖（sklearn + numpy）=========
SKLEARN_OK = True
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
except Exception:
    SKLEARN_OK = False
    np = None
    TfidfVectorizer = None
    LogisticRegression = None
    Pipeline = None
# =========================
# 配置
# =========================
CATEGORIES = ["教育", "健康", "生活", "娱乐", "游戏"]

CATEGORY_KEYWORDS = {
    "教育": ["学校", "老师", "学生", "课堂", "考试", "作业", "成绩", "学习", "课程", "教育", "考研", "高考", "大学", "中学", "小学"],
    "健康": ["健康", "医生", "医院", "药", "疾病", "治疗", "症状", "体检", "运动", "减肥", "血压", "糖尿病", "疫苗", "感染", "营养", "睡眠"],
    "生活": ["生活", "工作", "租房", "买房", "家庭", "孩子", "父母", "做饭", "出行", "购物", "价格", "快递", "工资", "消费", "社交"],
    "娱乐": ["娱乐", "电影", "明星", "综艺", "音乐", "演唱会", "电视剧", "粉丝", "八卦", "导演", "票房", "动画", "节目", "追剧"],
    "游戏": ["游戏", "玩家", "段位", "装备", "技能", "副本", "氪金", "手游", "端游", "英雄", "开黑", "电竞", "战队", "更新", "版本"],
}

STOPWORDS = {
    "的", "了", "和", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们", "以及", "一个", "这种", "那种", "这个", "那个",
    "因为", "所以", "但是", "如果", "然后", "而且", "并且", "或者", "进行", "需要", "可以", "不会", "没有", "还是", "就是",
    "非常", "一些", "自己", "可能", "已经", "目前", "同时", "还有", "什么", "怎么", "为什么", "不是", "也是",
}

# =========================
# 日志/结果文件
# =========================
LOG_FILENAME = "result_log.jsonl"
RESULT_DIR = os.path.join("examples", "result")
RESULT_PATH = os.path.join(RESULT_DIR, LOG_FILENAME)

# =========================
# 分类器缓存（方案A）
# =========================
_classifier_cache = {"model": None, "data_dir": None, "train_size": 0}
_classifier_dirty = True  # 目录切换/保存M后置 True


# =========================
# 通用工具
# =========================
def ensure_result_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u3000", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()


def _sentence_split(text: str):
    text = _clean_text(text)
    if not text:
        return []
    sents = re.split(r"[。！？!?；;]", text)
    return [s.strip() for s in sents if s.strip()]


def open_in_system(file_path: str):
    if not os.path.exists(file_path):
        messagebox.showwarning("提示", f"找不到文件：{file_path}")
        return
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)  # type: ignore
        elif platform.system() == "Darwin":
            subprocess.run(["open", file_path], check=False)
        else:
            subprocess.run(["xdg-open", file_path], check=False)
    except Exception as e:
        messagebox.showerror("错误", f"打开失败：{e}")


def read_one_jsonl(path: str):
    """
    兼容：
    - 整文件一个 JSON（多行也行）
    - 真 JSONL（多行多对象），取第一条可解析对象
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if not text:
        return "", ""

    # 先尝试整体 JSON
    try:
        obj = json.loads(text)
        title = obj.get("title") or obj.get("Title") or ""
        content = obj.get("content") or obj.get("Content") or obj.get("Document") or obj.get("document") or ""
        return title, content
    except Exception:
        pass

    # 再逐行 JSONL
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            title = obj.get("title") or obj.get("Title") or ""
            content = obj.get("content") or obj.get("Content") or obj.get("Document") or obj.get("document") or ""
            return title, content
        except Exception:
            continue

    raise ValueError(f"无法解析 JSON/JSONL：{path}")


def result_append(file, D_Mark, FileName, Title, KeyWord_HFWord, ClassLabel, NamedEntity, Abstract, Content):
    one_result = {
        "TimeStamp": [str(time.time())],
        "D_Mark": [D_Mark],
        "FileName": [FileName],
        "Title": [Title],
        "KeyWord_HFWord": [KeyWord_HFWord],
        "ClassLabel": [ClassLabel],
        "NamedEntity": [NamedEntity],
        "Abstract": [Abstract],
        "Content": [Content],
    }
    file.writelines(json.dumps(one_result, ensure_ascii=False) + "\n")


def _safe_get_listval(obj, key):
    v = obj.get(key)
    if isinstance(v, list) and v:
        return str(v[0])
    return "" if v is None else str(v)


def load_latest_record_for_file(file_name: str):
    """
    从 RESULT_PATH 中找该 FileName 的最新一条记录：
    - 优先 D_Mark = M 的最新
    - 若没有 M，则取 A 的最新
    """
    if not os.path.exists(RESULT_PATH):
        return None

    best_m, best_m_ts = None, -1.0
    best_a, best_a_ts = None, -1.0

    try:
        with open(RESULT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                if _safe_get_listval(obj, "FileName") != file_name:
                    continue

                dmark = _safe_get_listval(obj, "D_Mark")
                try:
                    ts = float(_safe_get_listval(obj, "TimeStamp"))
                except Exception:
                    ts = -1.0

                if dmark == "M":
                    if ts >= best_m_ts:
                        best_m_ts, best_m = ts, obj
                elif dmark == "A":
                    if ts >= best_a_ts:
                        best_a_ts, best_a = ts, obj
    except Exception:
        return None

    return best_m if best_m is not None else best_a


def export_log_to_csv(csv_path: str):
    ensure_result_dir()
    if not os.path.exists(RESULT_PATH):
        raise FileNotFoundError("result_log.jsonl 不存在")

    fields = ["TimeStamp", "D_Mark", "FileName", "Title", "KeyWord_HFWord", "ClassLabel", "NamedEntity", "Abstract", "Content"]

    with open(RESULT_PATH, "r", encoding="utf-8") as fin, open(csv_path, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            row = {k: _safe_get_listval(obj, k) for k in fields}
            writer.writerow(row)


# =========================
# 分词/关键词/高频词/实体
# =========================
def tokenize_for_sklearn(text: str):
    text = _clean_text(text)
    if not text:
        return []
    toks = []
    for w in jieba.lcut(text):
        w = w.strip()
        if not w or len(w) <= 1:
            continue
        if w.isdigit() or (w in STOPWORDS):
            continue
        toks.append(w)
    return toks


def extract_keywords_tfidf(content: str, topk: int = 10):
    content = _clean_text(content)
    return jieba.analyse.extract_tags(content, topK=topk) if content else []


def extract_keywords_textrank(content: str, topk: int = 10):
    content = _clean_text(content)
    if not content:
        return []
    try:
        return jieba.analyse.textrank(content, topK=topk, withWeight=False)
    except Exception:
        return []


def fuse_keywords(tfidf_kw, tr_kw, topk=10):
    out, seen = [], set()
    for w in tfidf_kw + tr_kw:
        if w not in seen:
            out.append(w)
            seen.add(w)
        if len(out) >= topk:
            break
    return out


def extract_high_freq_words(content: str, topk: int = 10):
    content = _clean_text(content)
    if not content:
        return []
    words = []
    for w in jieba.lcut(content):
        w = w.strip()
        if not w or len(w) <= 1:
            continue
        if w.isdigit() or (w in STOPWORDS):
            continue
        words.append(w)
    return [w for w, _ in Counter(words).most_common(topk)]


def extract_named_entities_light(content: str, topk: int = 10):
    content = _clean_text(content)
    if not content:
        return []
    ents = []
    for w, flag in pseg.cut(content):
        w = w.strip()
        if not w or len(w) <= 1:
            continue
        if flag in ("nr", "ns", "nt"):
            ents.append(w)
    out, seen = [], set()
    for e in ents:
        if e not in seen:
            out.append(e)
            seen.add(e)
        if len(out) >= topk:
            break
    return out


# =========================
# 分类：规则 + 可训练模型（TF-IDF + LR）
# =========================
def classify_by_keywords(title: str, content: str):
    title = _clean_text(title)
    content = _clean_text(content)
    text = (title + " " + content).strip()

    scores = {c: 0 for c in CATEGORIES}
    for c in CATEGORIES:
        for kw in CATEGORY_KEYWORDS[c]:
            if kw in title:
                scores[c] += 3
            if kw in text:
                scores[c] += 1

    best_c, best_s = max(scores.items(), key=lambda x: x[1])
    return best_c if best_s > 0 else "生活"


def _iter_corpus_files(data_dir: str):
    for name in os.listdir(data_dir):
        if name.endswith(".jsonl") and name != LOG_FILENAME:
            yield name


def build_or_get_classifier(data_dir: str):
    global _classifier_dirty

    if not SKLEARN_OK:
        return None, {"train_size": 0, "from_M": 0, "from_A": 0, "from_rule": 0}

    if (_classifier_cache["model"] is not None and
            _classifier_cache["data_dir"] == data_dir and
            not _classifier_dirty):
        return _classifier_cache["model"], _classifier_cache.get("stats", {"train_size": _classifier_cache.get("train_size", 0),
                                                                          "from_M": 0, "from_A": 0, "from_rule": 0})

    X_texts, y = [], []
    from_M = 0
    from_A = 0
    from_rule = 0

    for fn in _iter_corpus_files(data_dir):
        path = os.path.join(data_dir, fn)
        try:
            title, content = read_one_jsonl(path)
        except Exception:
            continue

        label = ""
        rec = load_latest_record_for_file(fn)

        if rec is not None:
            dmark = _safe_get_listval(rec, "D_Mark").strip()
            tmp = _safe_get_listval(rec, "ClassLabel").strip()
            if tmp in CATEGORIES:
                label = tmp
                if dmark == "M":
                    from_M += 1
                elif dmark == "A":
                    from_A += 1

        if not label:
            label = classify_by_keywords(title, content)
            from_rule += 1

        X_texts.append((title or "") + " " + (content or ""))
        y.append(label)

    if len(X_texts) < 10:
        return None, {"train_size": len(X_texts), "from_M": from_M, "from_A": from_A, "from_rule": from_rule}

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=tokenize_for_sklearn,
            token_pattern=None,
            max_features=5000
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="liblinear"
        ))
    ])

    model.fit(X_texts, y)

    stats = {"train_size": len(X_texts), "from_M": from_M, "from_A": from_A, "from_rule": from_rule}

    _classifier_cache["model"] = model
    _classifier_cache["data_dir"] = data_dir
    _classifier_cache["train_size"] = len(X_texts)
    _classifier_cache["stats"] = stats
    _classifier_dirty = False
    return model, stats



# =========================
# 摘要：TextRank + 信息密度增强 + MMR（3句）
# =========================
_GENERIC_PAT = re.compile(r"(本文|本研究|本报告|总体来看|综上|随着|近年来|目前来看|在.*背景下|我们认为)")
_NUM_PAT = re.compile(r"(\d+(\.\d+)?)(%|万元|亿|万|元|年|天|次|人|例)?")

def _info_density(sent: str, keywords_set: set):
    """
    信息密度：数字/实体/关键词覆盖/长度，惩罚泛句
    """
    s = sent.strip()
    if not s:
        return 0.0

    # 数字/量化
    num_cnt = len(_NUM_PAT.findall(s))

    # 实体（人/地/机构）
    ent_cnt = 0
    try:
        for w, flag in pseg.cut(s):
            if flag in ("nr", "ns", "nt"):
                ent_cnt += 1
    except Exception:
        ent_cnt = 0

    # 关键词覆盖
    toks = tokenize_for_sklearn(s)
    kw_hit = sum(1 for t in toks if t in keywords_set)

    # 长度（过短信息少）
    length_bonus = min(len(s) / 25.0, 2.0)  # 0~2

    # 泛句惩罚
    generic_penalty = 0.0
    if _GENERIC_PAT.search(s):
        generic_penalty = 1.0

    # 综合（可解释）
    score = (
        1.2 * num_cnt +
        1.0 * ent_cnt +
        0.8 * kw_hit +
        0.3 * length_bonus
        - 1.0 * generic_penalty
    )
    return score


def summarize_textrank_mmr(content: str, max_chars: int = 180, max_sents: int = 3):
    """
    三句摘要（默认）：TextRank（句子图+PageRank） -> 信息密度重排序 -> MMR去重选句
    - 若无 sklearn/numpy：退化为“主题+事实+补充”的规则抽取
    """
    content = _clean_text(content)
    if not content:
        return ""

    sents = _sentence_split(content)
    if not sents:
        return content[:max_chars]
    if len(sents) <= max_sents:
        return "。 ".join(sents)[:max_chars]

    # 限制句子数避免太慢
    sents = sents[:30]
    n = len(sents)

    # 关键词集合（给信息密度 & 主题覆盖用）
    tfidf_kw = extract_keywords_tfidf(content, topk=12)
    tr_kw = extract_keywords_textrank(content, topk=12)
    keywords = fuse_keywords(tfidf_kw, tr_kw, topk=12)
    kw_set = set(keywords)

    # ---- 降级：无 sklearn/numpy ----
    if not SKLEARN_OK:
        # 主题句：关键词覆盖最多
        cover = []
        for i, s in enumerate(sents):
            toks = tokenize_for_sklearn(s)
            cover.append((sum(1 for t in toks if t in kw_set), i))
        cover.sort(reverse=True)
        first = cover[0][1]

        # 事实句：包含数字/实体，且和主题不重复
        cand = []
        for i, s in enumerate(sents):
            if i == first:
                continue
            d = _info_density(s, kw_set)
            cand.append((d, i))
        cand.sort(reverse=True)

        picked = [first]
        for _, i in cand:
            if i not in picked:
                picked.append(i)
            if len(picked) >= max_sents:
                break

        picked = sorted(picked)
        abstract = "。 ".join(sents[i] for i in picked).strip()
        return abstract[:max_chars]

    # ---- TextRank 基础：句子TF-IDF + 相似度图 ----
    vec = TfidfVectorizer(tokenizer=tokenize_for_sklearn, token_pattern=None)
    X = vec.fit_transform(sents)  # (n, vocab)

    # 余弦相似（用点积近似足够；TF-IDF已归一化不严格，这里可用）
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0.0)

    # 行归一化得到转移矩阵
    row_sum = sim.sum(axis=1, keepdims=True) + 1e-12
    P = sim / row_sum

    # PageRank
    d = 0.85
    pr = np.ones(n) / n
    for _ in range(40):
        pr = (1 - d) / n + d * (P.T @ pr)

    # 信息密度增强（让摘要更“具体可核验”）
    dens = np.array([_info_density(s, kw_set) for s in sents], dtype=float)
    dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)  # 0~1
    final = pr * (1.0 + 0.9 * dens)

    # ---- MMR 选句：相关性 vs 多样性 ----
    # relevance: final 分
    # diversity: 与已选句子的最大相似度
    selected = []
    candidates = list(range(n))
    lam = 0.72

    # 先选最相关
    first = int(np.argmax(final))
    selected.append(first)
    candidates.remove(first)

    # 逐步选到 max_sents
    while len(selected) < max_sents and candidates:
        best_i, best_score = None, -1e9
        for i in candidates:
            rel = float(final[i])
            div = float(max(sim[i, j] for j in selected)) if selected else 0.0
            mmr = lam * rel - (1 - lam) * div
            if mmr > best_score:
                best_score, best_i = mmr, i
        selected.append(best_i)
        candidates.remove(best_i)

    selected = sorted(selected)  # 按原文顺序
    abstract = "。 ".join(sents[i] for i in selected).strip()
    return abstract[:max_chars]


# =========================
# 统一分析入口
# =========================
def analyze_doc(title: str, content: str, data_dir: str):
    title = title or ""
    content = content or ""

    # 关键词融合
    tfidf_kw = extract_keywords_tfidf(content, topk=10)
    tr_kw = extract_keywords_textrank(content, topk=10)
    keywords = fuse_keywords(tfidf_kw, tr_kw, topk=10)

    hf_words = extract_high_freq_words(content, topk=10)
    entities = extract_named_entities_light(content, topk=10)

    # 三句摘要（升级版）
    abstract = summarize_textrank_mmr(content, max_chars=220, max_sents=3)

    # 分类：优先 ML，否则规则兜底
    # 分类：优先 ML，否则规则兜底
    cls = ""
    if SKLEARN_OK:
        model, stats = build_or_get_classifier(data_dir)  # ✅接收两个返回值
        if model is not None:
            try:
                cls = model.predict([_clean_text(title) + " " + _clean_text(content)])[0]  # ✅用 model 来 predict
            except Exception:
                cls = ""

    if cls not in CATEGORIES:
        cls = classify_by_keywords(title, content)

    kw_str = ",".join(keywords)
    hf_str = ",".join(hf_words)
    key_hf = f"{kw_str},|{hf_str}" if (kw_str or hf_str) else ",|"

    named_entity = "，".join(entities)
    return key_hf, cls, named_entity, abstract


# =========================
# GUI
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DocumentAnalyzer - 作者谢佳琪")
        self.geometry("1180x720")

        self.data_dir = os.path.abspath("examples")
        self.files = []
        self.current_file = None
        self.dirty = False

        ensure_result_dir()

        self._build_ui()
        self._load_dir(self.data_dir)

        if not SKLEARN_OK:
            messagebox.showwarning(
                "依赖提示",
                "未检测到 sklearn/numpy：摘要会使用降级抽取逻辑，分类将使用规则为主。\n"
                "建议：pip install scikit-learn numpy"
            )

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.dir_var = tk.StringVar(value=self.data_dir)
        ttk.Label(top, text="语料目录:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.dir_var, width=55).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="选择目录", command=self.choose_dir).pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="训练分类器", command=self.train_classifier).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="自动分析当前", command=self.analyze_current).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="批量分析并保存(A)", command=self.analyze_all_and_save).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="保存人工更改(M)", command=self.save_manual).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="打开结果日志", command=self.open_result_log).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="导出CSV", command=self.export_csv).pack(side=tk.LEFT, padx=6)

        main = ttk.Frame(self, padding=8)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 左：文件列表
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="文件列表（点击选择；自动回显最新A/M）").pack(anchor="w")
        self.listbox = tk.Listbox(left, width=35, height=34)
        self.listbox.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_file)

        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        # 右：内容区
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        doc_frame = ttk.LabelFrame(right, text="原文（title + content）", padding=8)
        doc_frame.pack(fill=tk.BOTH, expand=True)

        self.title_var = tk.StringVar(value="")
        ttk.Label(doc_frame, text="Title:").pack(anchor="w")
        ttk.Entry(doc_frame, textvariable=self.title_var).pack(fill=tk.X, pady=(0, 6))

        ttk.Label(doc_frame, text="Content:").pack(anchor="w")
        self.content_text = tk.Text(doc_frame, height=12, wrap=tk.WORD)
        self.content_text.pack(fill=tk.BOTH, expand=True)

        result_frame = ttk.LabelFrame(right, text="分析结果（可人工编辑；保存后生成M版本）", padding=8)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        info = ttk.Frame(result_frame)
        info.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.record_info_var = tk.StringVar(value="日志回显：无")
        ttk.Label(info, textvariable=self.record_info_var).pack(side=tk.LEFT)

        ttk.Label(result_frame, text="KeyWord_HFWord（关键词...,|高频词...）").grid(row=1, column=0, sticky="w")
        self.keyhf_var = tk.StringVar(value="")
        e_key = ttk.Entry(result_frame, textvariable=self.keyhf_var)
        e_key.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        e_key.bind("<KeyRelease>", self._mark_dirty)

        ttk.Label(result_frame, text="ClassLabel").grid(row=2, column=0, sticky="w")
        self.class_var = tk.StringVar(value="生活")
        self.class_combo = ttk.Combobox(result_frame, textvariable=self.class_var, values=CATEGORIES, state="readonly")
        self.class_combo.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        self.class_combo.bind("<<ComboboxSelected>>", self._mark_dirty)

        ttk.Label(result_frame, text="NamedEntity（中文逗号分隔）").grid(row=3, column=0, sticky="w")
        self.entity_var = tk.StringVar(value="")
        e_ent = ttk.Entry(result_frame, textvariable=self.entity_var)
        e_ent.grid(row=3, column=1, sticky="ew", padx=6, pady=4)
        e_ent.bind("<KeyRelease>", self._mark_dirty)

        ttk.Label(result_frame, text="Abstract（三句摘要）").grid(row=4, column=0, sticky="nw")
        self.abstract_text = tk.Text(result_frame, height=7, wrap=tk.WORD)
        self.abstract_text.grid(row=4, column=1, sticky="ew", padx=6, pady=4)
        self.abstract_text.bind("<KeyRelease>", self._mark_dirty)

        result_frame.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self, textvariable=self.status_var, padding=6).pack(side=tk.BOTTOM, fill=tk.X)

    def _mark_dirty(self, event=None):
        self.dirty = True
        if self.current_file:
            self.status_var.set(f"当前文件：{self.current_file}（有未保存修改）")

    def choose_dir(self):
        d = filedialog.askdirectory(title="选择语料目录")
        if d:
            self._load_dir(d)

    def _load_dir(self, d):
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            messagebox.showerror("错误", f"不是目录：{d}")
            return

        self.data_dir = d
        self.dir_var.set(d)

        files = [name for name in os.listdir(d) if name.endswith(".jsonl") and name != LOG_FILENAME]

        def _key(x):
            base = os.path.splitext(x)[0]
            return (0, int(base)) if base.isdigit() else (1, x)

        files.sort(key=_key)

        self.files = files
        self.listbox.delete(0, tk.END)
        for n in self.files:
            self.listbox.insert(tk.END, n)

        self.current_file = None
        self._clear_right_panel()
        self.record_info_var.set("日志回显：无")
        self.status_var.set(f"已加载目录：{d}，共 {len(files)} 个文件")

        global _classifier_dirty
        _classifier_dirty = True

    def _clear_right_panel(self):
        self.title_var.set("")
        self.content_text.delete("1.0", tk.END)
        self.keyhf_var.set("")
        self.class_var.set("生活")
        self.entity_var.set("")
        self.abstract_text.delete("1.0", tk.END)
        self.dirty = False

    def on_select_file(self, event):
        if not self.listbox.curselection():
            return

        if self.dirty and self.current_file:
            if not messagebox.askyesno("提示", "当前结果区有未保存修改，切换文件会丢失这些修改。继续切换吗？"):
                return

        idx = self.listbox.curselection()[0]
        name = self.files[idx]
        self.current_file = name
        self.dirty = False

        path = os.path.join(self.data_dir, name)
        try:
            title, content = read_one_jsonl(path)
        except Exception as e:
            messagebox.showerror("错误", f"读取失败：{e}")
            return

        self.title_var.set(title)
        self.content_text.delete("1.0", tk.END)
        self.content_text.insert(tk.END, content)

        rec = load_latest_record_for_file(name)
        if rec is None:
            self.keyhf_var.set("")
            self.class_var.set("生活")
            self.entity_var.set("")
            self.abstract_text.delete("1.0", tk.END)
            self.record_info_var.set("日志回显：无（可点“自动分析当前”生成A，再保存M）")
        else:
            self.record_info_var.set(
                f"日志回显：D_Mark={_safe_get_listval(rec,'D_Mark')}  TimeStamp={_safe_get_listval(rec,'TimeStamp')}"
            )
            self.keyhf_var.set(_safe_get_listval(rec, "KeyWord_HFWord"))
            cls = _safe_get_listval(rec, "ClassLabel") or "生活"
            self.class_var.set(cls if cls in CATEGORIES else "生活")
            self.entity_var.set(_safe_get_listval(rec, "NamedEntity"))
            self.abstract_text.delete("1.0", tk.END)
            self.abstract_text.insert(tk.END, _safe_get_listval(rec, "Abstract"))

        self.status_var.set(f"当前文件：{name}（已载入原文 + 回显最新结果）")

    def train_classifier(self):
        if not SKLEARN_OK:
            messagebox.showwarning("提示", "未安装 scikit-learn/numpy，无法训练。\n请先 pip install scikit-learn numpy")
            return

        model, stats = build_or_get_classifier(self.data_dir)
        if model is None:
            messagebox.showwarning("提示", f"可用训练数据不足（当前样本：{stats.get('train_size', 0)}）。")
            return

        messagebox.showinfo(
            "完成",
            "分类器训练完成（TF-IDF + LogisticRegression）。\n"
            f"训练样本数：{stats['train_size']}\n"
            f"标签来源：M={stats['from_M']}  A={stats['from_A']}  规则={stats['from_rule']}\n"
            "后续分析会优先用模型预测。"
        )

    def analyze_current(self):
        if not self.current_file:
            messagebox.showwarning("提示", "请先在左侧选择一个文件")
            return

        title = self.title_var.get()
        content = self.content_text.get("1.0", tk.END).strip()
        key_hf, cls, ents, abstract = analyze_doc(title, content, self.data_dir)

        self.keyhf_var.set(key_hf)
        self.class_var.set(cls)
        self.entity_var.set(ents)
        self.abstract_text.delete("1.0", tk.END)
        self.abstract_text.insert(tk.END, abstract)

        self.dirty = True
        self.record_info_var.set("日志回显：无（当前为界面新分析结果，未保存）")
        self.status_var.set(f"已自动分析：{self.current_file}（未保存，建议保存M）")

    def save_manual(self):
        if not self.current_file:
            messagebox.showwarning("提示", "请先选择文件")
            return

        title = self.title_var.get()
        content = self.content_text.get("1.0", tk.END).strip()

        key_hf = self.keyhf_var.get().strip()
        cls = self.class_var.get().strip()
        ents = self.entity_var.get().strip()
        abstract = self.abstract_text.get("1.0", tk.END).strip()

        if cls not in CATEGORIES:
            messagebox.showwarning("提示", f"ClassLabel 必须是：{CATEGORIES}")
            return

        ensure_result_dir()
        try:
            with open(RESULT_PATH, "a", encoding="utf-8") as f:
                result_append(f, "M", self.current_file, title, key_hf, cls, ents, abstract, content)
        except Exception as e:
            messagebox.showerror("错误", f"保存失败：{e}")
            return

        global _classifier_dirty
        _classifier_dirty = True

        self.dirty = False
        rec = load_latest_record_for_file(self.current_file)
        if rec:
            self.record_info_var.set(
                f"日志回显：D_Mark={_safe_get_listval(rec,'D_Mark')}  TimeStamp={_safe_get_listval(rec,'TimeStamp')}"
            )
        self.status_var.set(f"已追加保存（M）：{self.current_file}")
        messagebox.showinfo("完成", "已保存人工更改结果（D_Mark=M，追加写入日志）")

    def analyze_all_and_save(self):
        if not self.files:
            messagebox.showwarning("提示", "当前目录没有可分析的 .jsonl 文件")
            return

        if not messagebox.askyesno("确认", f"将对目录下 {len(self.files)} 个文件自动分析并追加写入日志（D_Mark=A）。继续吗？"):
            return

        ensure_result_dir()
        ok, fail = 0, 0
        errors = []

        for name in self.files:
            path = os.path.join(self.data_dir, name)
            try:
                title, content = read_one_jsonl(path)
                key_hf, cls, ents, abstract = analyze_doc(title, content, self.data_dir)
                with open(RESULT_PATH, "a", encoding="utf-8") as f:
                    result_append(f, "A", name, title, key_hf, cls, ents, abstract, content)
                ok += 1
                if ok % 10 == 0:
                    self.status_var.set(f"批量分析中... {ok}/{len(self.files)}")
                    self.update_idletasks()
            except Exception as e:
                fail += 1
                if len(errors) < 3:
                    errors.append(f"{name}: {repr(e)}")

        msg = f"批量分析完成：成功 {ok}，失败 {fail}\n日志：{RESULT_PATH}"
        if errors:
            msg += "\n\n失败示例（前3条）：\n" + "\n".join(errors)

        self.status_var.set(f"批量分析完成：成功 {ok}，失败 {fail}（结果已追加写入日志）")
        messagebox.showinfo("完成", msg)

    def open_result_log(self):
        ensure_result_dir()
        open_in_system(RESULT_PATH)

    def export_csv(self):
        ensure_result_dir()
        default_csv = os.path.join(RESULT_DIR, "result_log.csv")
        csv_path = filedialog.asksaveasfilename(
            title="导出CSV",
            defaultextension=".csv",
            initialfile=os.path.basename(default_csv),
            initialdir=os.path.dirname(default_csv),
            filetypes=[("CSV files", "*.csv")]
        )
        if not csv_path:
            return
        try:
            export_log_to_csv(csv_path)
        except Exception as e:
            messagebox.showerror("错误", f"导出失败：{e}")
            return
        messagebox.showinfo("完成", f"已导出CSV：{csv_path}")
        self.status_var.set(f"已导出CSV：{csv_path}")


if __name__ == "__main__":
    ensure_result_dir()
    app = App()
    app.mainloop()
