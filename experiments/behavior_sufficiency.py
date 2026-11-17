"""实验二：三种证据条件下的模型回答行为分析。

设计
----
- 30 个虚构问题 × 3 种证据条件（充分 / 不足 / 无证据-无关文本）= 90 次生成。
- 复用实验一的模型加载函数与 Prompt 模板（不修改实验一文件）。
- 确定性解码 do_sample=False（贪心），max_new_tokens=128，记录完整输入与输出。
- 自动分类为四类（规则仅是初筛，全部原文保留供人工复核）：
    evidence_supported   有证据支持（仅限充分条件且答案含金标）
    unfounded_specific   无依据确定回答（给出具体实体但非"充分条件+金标"）
    explicit_uncertainty 明确表示无法确定
    other                其他 / 无法自动判断
- 关键规则：非充分条件下即使答案碰巧等于金标，也只算"猜对"，计入
  unfounded_specific，绝不计入 evidence_supported。
- 本脚本不训练探针、不做 Activation Patching。

运行（Windows PowerShell，借用环境只读）：
    & 'D:\\workspace\\resume\\qlora-text2sql\\.venv\\Scripts\\python.exe' `
        'D:\\workspace\\rag-internal-mechanisms\\experiments\\behavior_sufficiency.py'
"""

import argparse
import csv
import json
import platform
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

# 复用实验一脚本中的组件（同目录导入）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hidden_states_sufficiency import (  # noqa: E402
    DEFAULT_MODEL,
    PROMPT_TEMPLATE,
    check_model_files,
    load_model_and_tokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "behavior_conditions_v1.json"
CONDITIONS = ["sufficient", "insufficient", "none"]
CONDITION_ZH = {"sufficient": "证据充分", "insufficient": "证据不足", "none": "无证据"}
LABELS = ["evidence_supported", "unfound_specific", "explicit_uncertainty", "other"]
LABEL_ZH = {
    "evidence_supported": "有证据支持",
    "unfound_specific": "无依据确定回答",
    "explicit_uncertainty": "明确表示无法确定",
    "other": "其他/无法自动判断",
}

# 不确定性 / 拒答表述（命中即视为"明确表示无法确定"的候选）
UNCERTAINTY_MARKERS = [
    "无法确定", "不能确定", "无法判断", "无法从", "无从判断", "不知道",
    "没有提到", "未提到", "没提到", "没有提及", "未提及", "没有说明", "未说明",
    "没有指出", "未指出", "没有提供", "未提供", "没有相关信息", "没有直接",
    "并未说明", "并没有说明", "只提到", "仅提到", "只是提到", "没有涉及",
    "并未涉及", "证据中没", "证据未", "不足以", "无法回答", "不能回答",
    "尚不清楚", "无法得出", "没有明确",
]


def contains_candidate(text: str, cand: str) -> bool:
    """候选实体匹配；数字候选用数字边界，避免 2016 误含 16 之类的子串误报。"""
    if cand.isdigit():
        return re.search(rf"(?<!\d){re.escape(cand)}(?!\d)", text) is not None
    return cand in text


def classify(answer: str, condition: str, gold: str, candidates: list) -> tuple:
    """规则化四分类，返回 (label, rule_note)。规则顺序与理由见注释。"""
    # 1) 找最早出现的不确定性表述位置
    marker_pos, marker_hit = None, None
    for m in UNCERTAINTY_MARKERS:
        p = answer.find(m)
        if p != -1 and (marker_pos is None or p < marker_pos):
            marker_pos, marker_hit = p, m
    # 2) 找不确定性表述之后是否仍给出具体候选（"没提到X，但是Y"这类混合回答）。
    #    注意：与表述同处一个小句的实体是表述的一部分（如"只提到清华大学"），
    #    因此只检查该表述之后第一个句读符号再往后的内容。
    candidates_in = [c for c in candidates if contains_candidate(answer, c)]
    if marker_pos is not None:
        scan_from = marker_pos + len(marker_hit)
        boundary = None
        for p in "，。！？；、,;:：":
            idx = answer.find(p, scan_from)
            if idx != -1 and (boundary is None or idx < boundary):
                boundary = idx
        after_text = answer[boundary:] if boundary is not None else ""
        after = [c for c in candidates_in if contains_candidate(after_text, c)]
        if after:
            return "other", f"混合：含『{marker_hit}』但其后仍给出具体答案 {after}"
        return "explicit_uncertainty", f"命中不确定性表述『{marker_hit}』"
    # 3) 无不确定性表述
    if condition == "sufficient" and contains_candidate(answer, gold):
        return "evidence_supported", "充分条件且答案包含金标"
    if candidates_in:
        # 给出了具体实体但不是"充分+金标"：包括不足/无证据条件的任何具体作答，
        # 以及充分条件下的答非金标（猜对也不算有证据支持，规则见模块注释）
        return "unfound_specific", f"给出具体答案候选 {candidates_in}（金标={gold}）"
    return "other", "无不确定性表述且无候选实体命中，需人工复核"


def generate_answer(model, tokenizer, evidence: str, question: str, max_new_tokens: int):
    """构造与实验一完全一致格式的 prompt，贪心解码生成回答。"""
    content = PROMPT_TEMPLATE.format(evidence=evidence, question=question)
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 确定性解码
            pad_token_id=tokenizer.eos_token_id,
        )
    n_prompt = inputs.input_ids.shape[1]
    new_tokens = output[0, n_prompt:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    truncated = len(new_tokens) >= max_new_tokens
    del output, inputs, new_tokens
    torch.cuda.empty_cache()
    return prompt, answer, peak_gb, truncated


def main() -> None:
    parser = argparse.ArgumentParser(description="三种证据条件下的回答行为分析")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="调试用：只跑前 N 题")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    questions = dataset["questions"]
    if args.limit:
        questions = questions[: args.limit]
    pool = dataset["irrelevant_pool"]

    print("[1/4] 检查本地模型文件 ...")
    check_model_files(args.model)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请用带 CUDA 版 PyTorch 的解释器运行")

    print("[2/4] 加载模型（bfloat16, local_files_only=True）...")
    tokenizer, model = load_model_and_tokenizer(args.model, torch.bfloat16)

    print(f"[3/4] 开始生成：{len(questions)} 题 × {len(CONDITIONS)} 条件 = "
          f"{len(questions) * len(CONDITIONS)} 次 ...")
    rows = []
    total = len(questions) * len(CONDITIONS)
    done = 0
    max_peak = 0.0
    for q in questions:
        for cond in CONDITIONS:
            evidence = q["evidences"][cond]
            if isinstance(evidence, int):  # none 条件存的是无关文本池下标
                evidence = pool[evidence]
            prompt, answer, peak_gb, truncated = generate_answer(
                model, tokenizer, evidence, q["question"], args.max_new_tokens
            )
            max_peak = max(max_peak, peak_gb)
            label, note = classify(answer, cond, q["gold_answer"], q["candidates"])
            done += 1
            rows.append({
                "question_id": q["id"],
                "type": q["type"],
                "condition": cond,
                "condition_zh": CONDITION_ZH[cond],
                "question": q["question"],
                "evidence": evidence,
                "gold_answer": q["gold_answer"],
                "answer": answer,
                "label": label,
                "label_zh": LABEL_ZH[label],
                "rule_note": note,
                "answer_chars": len(answer),
                "truncated": truncated,
                "peak_gb": round(peak_gb, 3),
            })
            print(f"  [{done:2d}/{total}] {q['id']} {cond:<12} → {label}{'（截断）' if truncated else ''}")

    # ---- 汇总统计 ----
    print("[4/4] 汇总与保存 ...")
    summary = []
    for cond in CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond]
        n = len(sub)
        counts = {lab: sum(1 for r in sub if r["label"] == lab) for lab in LABELS}
        summary.append({
            "condition": cond,
            "condition_zh": CONDITION_ZH[cond],
            "n": n,
            **counts,
            "pct_evidence_supported": round(counts["evidence_supported"] / n * 100, 1),
            "pct_unfound_specific": round(counts["unfound_specific"] / n * 100, 1),
            "pct_explicit_uncertainty": round(counts["explicit_uncertainty"] / n * 100, 1),
            "pct_other": round(counts["other"] / n * 100, 1),
        })

    record = {
        "experiment": "behavior_sufficiency_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiments/behavior_sufficiency.py",
        "dataset": str(args.dataset),
        "model_path": str(args.model),
        "decoding": "greedy (do_sample=False)",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "max_peak_gb": round(max_peak, 3),
        "classification_note": (
            "规则自动分类仅为初筛；『猜对答案』一律计入 unfound_specific，"
            "不算有证据支持。全部回答保留原文，须人工复核后引用。"
        ),
        "summary": summary,
        "generations": rows,
    }

    out_json = PROJECT_ROOT / "results" / "behavior_v1_generations.json"
    out_rows = PROJECT_ROOT / "results" / "behavior_v1_rows.csv"
    out_summary = PROJECT_ROOT / "results" / "behavior_v1_summary.csv"
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_rows.open("w", newline="", encoding="utf-8-sig") as f:  # BOM 便于 Excel 打开中文
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with out_summary.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("=" * 66)
    for s in summary:
        print(f"{s['condition_zh']}(n={s['n']}): 有证据支持 {s['evidence_supported']} | "
              f"无依据回答 {s['unfound_specific']} | 拒答 {s['explicit_uncertainty']} | "
              f"其他 {s['other']}")
    print(f"峰值显存 {max_peak:.2f} GB")
    print(f"已保存: {out_json.name}, {out_rows.name}, {out_summary.name}")
    print("=" * 66)


if __name__ == "__main__":
    main()
