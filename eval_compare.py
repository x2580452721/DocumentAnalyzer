import os, json, re
from collections import defaultdict
import jieba

# 可选依赖
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

CATEGORIES = ["教育", "健康", "生活", "娱乐", "游戏"]

CATEGORY_KEYWORDS = {
    "教育": ["学校","老师","学生","课堂","考试","作业","成绩","学习","课程","教育","考研","高考","大学","中学","小学"],
    "健康": ["健康","医生","医院","药","疾病","治疗","症状","体检","运动","减肥","血压","糖尿病","疫苗","感染","营养","睡眠"],
    "生活": ["生活","工作","租房","买房","家庭","孩子","父母","做饭","出行","购物","价格","快递","工资","消费","社交"],
    "娱乐": ["娱乐","电影","明星","综艺","音乐","演唱会","电视剧","粉丝","八卦","导演","票房","动画","节目","追剧"],
    "游戏": ["游戏","玩家","段位","装备","技能","副本","氪金","手游","端游","英雄","开黑","电竞","战队","更新","版本"],
}

STOPWORDS = set(["的","了","和","是","我","你","他","她","它","我们","你们","他们","以及","一个","这种","那种","这个","那个",
                 "因为","所以","但是","如果","然后","而且","并且","或者","进行","需要","可以","不会","没有","还是","就是",
                 "非常","一些","自己","可能","已经","目前","同时","还有","什么","怎么","为什么","不是","也是"])

RESULT_PATH = os.path.join("examples", "result", "result_log.jsonl")

def _clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\u3000"," ").replace("\r"," ").replace("\n"," ")
    s = re.sub(r"\s+"," ",s).strip()
    return s

def tokenize(text: str):
    text = _clean_text(text)
    toks = []
    for w in jieba.lcut(text):
        w = w.strip()
        if not w or len(w) <= 1:
            continue
        if w.isdigit() or w in STOPWORDS:
            continue
        toks.append(w)
    return toks

def read_doc(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if not text: return "", ""
    # JSON
    try:
        obj = json.loads(text)
        title = obj.get("title") or obj.get("Title") or ""
        content = obj.get("content") or obj.get("Content") or obj.get("Document") or obj.get("document") or ""
        return title, content
    except Exception:
        pass
    # JSONL
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
            title = obj.get("title") or obj.get("Title") or ""
            content = obj.get("content") or obj.get("Content") or obj.get("Document") or obj.get("document") or ""
            return title, content
        except Exception:
            continue
    return "", ""

def safe_listval(obj, k):
    v = obj.get(k, "")
    if isinstance(v, list) and v: return str(v[0])
    return "" if v is None else str(v)

def latest_m_labels():
    """取每个 FileName 最新的 M 标签"""
    latest = {}
    if not os.path.exists(RESULT_PATH):
        return latest
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if safe_listval(obj, "D_Mark") != "M":
                continue
            fn = safe_listval(obj, "FileName")
            ts = safe_listval(obj, "TimeStamp")
            try:
                ts = float(ts)
            except Exception:
                ts = -1
            lab = safe_listval(obj, "ClassLabel").strip()
            if lab not in CATEGORIES:
                continue
            if (fn not in latest) or (ts >= latest[fn][0]):
                latest[fn] = (ts, lab)
    return {fn: lab for fn, (ts, lab) in latest.items()}

def classify_by_keywords(title, content):
    title = _clean_text(title); content = _clean_text(content)
    text = (title + " " + content).strip()
    scores = {c: 0 for c in CATEGORIES}
    for c in CATEGORIES:
        for kw in CATEGORY_KEYWORDS[c]:
            if kw in title: scores[c] += 3
            if kw in text:  scores[c] += 1
    best_c, best_s = max(scores.items(), key=lambda x: x[1])
    return best_c if best_s > 0 else "生活"

def main():
    labels = latest_m_labels()
    if len(labels) < 10:
        print(f"[WARN] M 标注太少：{len(labels)} 条，建议>=10")
    # 组装数据
    X, y = [], []
    for fn, lab in labels.items():
        doc_path = os.path.join("examples", fn)  # 若语料目录不是 examples，自行改这里
        if not os.path.exists(doc_path):
            # 尝试在当前目录下找
            doc_path = fn
        title, content = read_doc(doc_path)
        X.append((title or "") + " " + (content or ""))
        y.append(lab)

    X = np.array(X); y = np.array(y)
    print("M 样本数:", len(y))

    # 规则 baseline（对全体直接测）
    y_rule = [classify_by_keywords("", x) for x in X]
    print("[Rule] Acc:", accuracy_score(y, y_rule), "Macro-F1:", f1_score(y, y_rule, average="macro"))

    # 5折CV 训练模型
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        model = Pipeline([
            ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
        ])
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    print("[TFIDF+LR 5fold] Acc:", float(np.mean(accs)), "Macro-F1:", float(np.mean(f1s)))

if __name__ == "__main__":
    main()
