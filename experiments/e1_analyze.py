"""E1：受控事实关系与回答行为分析 —— 回答判定与统计报告。

判定规则（保守，不强行判对错）：
1. 优先取回答中最后一个「答案：」之后、首个换行/句读之前的一段作为作答段。
2. 作答段中恰好只出现一个候选核心词 → 判为该候选。
3. 作答段只出现字母（A/B 或 候选A/B）→ 按字母判（记录 method=letter）。
4. 两个候选核心词都出现 → ambiguous（无法自动识别，单独记录）。
5. 无候选词且命中不确定性表述 → refusal；否则 unrecognized。
两种 unclear 均不计入准确率分子分母，全部原文保留供人工复核。

统计：总体 / 四条件 / 前半句vs后半句（含两比例 z 检验）/ 首句机构倾向 /
候选A倾向 / 分领域 / 错误与无法识别样本明细。

运行：
    & 'D:\\workspace\\resume\\qlora-text2sql\\.venv\\Scripts\\python.exe' `
        'D:\\workspace\\rag-internal-mechanisms\\experiments\\e1_analyze.py' `
        --responses 'D:\\workspace\\rag-internal-mechanisms\\results\\e1_responses.json'
"""

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES = PROJECT_ROOT / "results" / "e1_responses.json"
CONDITIONS = ["A1", "A2", "B1", "B2"]

UNCERTAINTY_MARKERS = [
    "无法确定", "不能确定", "无法判断", "无从判断", "不知道", "无法回答", "不能回答",
    "没有提到", "未提到", "没提到", "没有提及", "未提及", "没有说明", "未说明",
    "没有提供", "未提供", "不足以", "尚不清楚", "无法得出",
]


def answer_span(raw: str) -> str:
    """取最后一个「答案：」之后到首个换行/句读前的片段；无标记则用全文。"""
    pos = raw.rfind("答案")
    if pos != -1:
        seg = raw[pos:]
        colon = seg.find("：") if "：" in seg[:4] else seg.find(":")
        if colon != -1 and colon <= 3:
            seg = seg[colon + 1 :]
        for cut in ["\n", "。", "；", "，", "！", "？", "（", "("]:
            idx = seg.find(cut)
            if idx != -1:
                seg = seg[:idx]
        return seg.strip()
    return raw.strip()


NEGATION_MARKERS = ["不是", "而不是", "并非", "而非", "并不", "错误的", "错答"]


def letter_hit(span: str) -> str | None:
    """作答段只由字母选项构成时按字母判。"""
    s = span.replace(" ", "").replace(":", "").replace("：", "")
    if re.fullmatch(r"(候选|选项)?A.*", s) and "B" not in s:
        return "A"
    if re.fullmatch(r"(候选|选项)?B.*", s) and "A" not in s:
        return "B"
    return None


def classify(item: dict, raw: str) -> dict:
    span = answer_span(raw)
    a_hit, b_hit = item["core_a"] in span, item["core_b"] in span
    if any(n in span for n in NEGATION_MARKERS):
        # 作答段含否定表述（如「不是X，而是Y」截断后只剩被否定候选）：不强行判，转人工复核
        pred, method, status = None, "negation_in_span", "unrecognized"
    elif a_hit and not b_hit:
        pred, method, status = "A", "name", "scored"
    elif b_hit and not a_hit:
        pred, method, status = "B", "name", "scored"
    elif not a_hit and not b_hit:
        letter = letter_hit(span)
        if letter:
            pred, method, status = letter, "letter", "scored"
        elif any(m in raw for m in UNCERTAINTY_MARKERS):
            pred, method, status = None, "uncertainty_marker", "refusal"
        else:
            pred, method, status = None, "no_candidate_found", "unrecognized"
    else:
        pred, method, status = None, "both_candidates_in_span", "ambiguous"
    correct = (pred is not None and pred == ("A" if item["gold_is_a"] else "B")) if status == "scored" else None
    # 前半句机构：前半句小句中包含哪个候选核心词
    first_core = "A" if item["core_a"] in item["evidence"].split("；")[0] else "B"
    picked_first = (pred == first_core) if pred is not None else None
    return {
        "answer_span": span,
        "predicted": pred,
        "predicted_name": (item["cand_a"] if pred == "A" else item["cand_b"]) if pred else None,
        "method": method,
        "status": status,
        "correct": correct,
        "first_half_core": first_core,
        "picked_first_half": picked_first,
    }


def two_prop_z_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """两比例 z 检验（合并方差正态近似），返回 (z, 双侧 p)。"""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_val


def acc(rows: list) -> tuple[int, int, float]:
    scored = [r for r in rows if r["eval"]["status"] == "scored"]
    n_ok = sum(1 for r in scored if r["eval"]["correct"])
    n = len(scored)
    return n_ok, n, (n_ok / n if n else float("nan"))


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 统计分析")
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--out-prefix", type=str, default="e1")
    args = parser.parse_args()

    data = json.loads(args.responses.read_text(encoding="utf-8"))
    rows = []
    for g in data["generations"]:
        ev = classify(g, g["answer_raw"])
        rows.append({**{k: g[k] for k in [
            "item_id", "group_id", "domain", "condition", "question", "evidence",
            "cand_a", "cand_b", "core_a", "core_b", "gold_answer", "gold_is_a",
            "gold_half", "cand_order", "answer_raw"]}, "eval": ev})

    # ---- 统计 ----
    n_total = len(rows)
    ok, n_scored, overall = acc(rows)
    unclear = [r for r in rows if r["eval"]["status"] != "scored"]

    by_cond = {}
    for c in CONDITIONS:
        sub = [r for r in rows if r["condition"] == c]
        c_ok, c_n, c_acc = acc(sub)
        first_picks = [r["eval"]["picked_first_half"] for r in sub if r["eval"]["picked_first_half"] is not None]
        by_cond[c] = {
            "n": len(sub), "correct": c_ok, "scored": c_n, "accuracy": c_acc,
            "unclear": len(sub) - c_n,
            "first_half_pick_rate": (sum(first_picks) / len(first_picks)) if first_picks else None,
        }

    first_rows = [r for r in rows if r["gold_half"] == "first"]
    second_rows = [r for r in rows if r["gold_half"] == "second"]
    f_ok, f_n, f_acc = acc(first_rows)
    s_ok, s_n, s_acc = acc(second_rows)
    z, p = two_prop_z_test(f_ok, f_n, s_ok, s_n)

    all_picks = [r["eval"]["picked_first_half"] for r in rows if r["eval"]["picked_first_half"] is not None]
    first_pick_rate = sum(all_picks) / len(all_picks) if all_picks else None
    a_picks = [r for r in rows if r["eval"]["predicted"] == "A"]
    a_pick_rate = len(a_picks) / n_scored if n_scored else None

    by_domain = {}
    for d in sorted({r["domain"] for r in rows}):
        d_ok, d_n, d_acc = acc([r for r in rows if r["domain"] == d])
        by_domain[d] = {"n": len([r for r in rows if r["domain"] == d]), "scored": d_n, "correct": d_ok, "accuracy": d_acc}

    errors = [r for r in rows if r["eval"]["status"] == "scored" and not r["eval"]["correct"]]

    # ---- 保存 ----
    out_dir = PROJECT_ROOT / "results"
    record = {
        "experiment": "e1_position_bias_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiments/e1_analyze.py",
        "responses_file": str(args.responses),
        "overall": {"n_total": n_total, "scored": n_scored, "correct": ok, "accuracy": overall,
                    "unclear": len(unclear)},
        "by_condition": by_cond,
        "by_gold_half": {
            "first": {"n": len(first_rows), "scored": f_n, "correct": f_ok, "accuracy": f_acc},
            "second": {"n": len(second_rows), "scored": s_n, "correct": s_ok, "accuracy": s_acc},
            "z": round(z, 3), "p_two_sided": round(p, 4),
        },
        "position_tendency": {
            "first_half_pick_rate": first_pick_rate,
            "candidate_A_pick_rate": a_pick_rate,
            "n_picks": len(all_picks),
        },
        "by_domain": by_domain,
        "rows": rows,
    }
    out_json = out_dir / f"{args.out_prefix}_scored.json"
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = [{**{k: v for k, v in r.items() if k != "eval"}, **r["eval"]} for r in rows]
    out_csv = out_dir / f"{args.out_prefix}_rows.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        w.writeheader()
        w.writerows(flat_rows)

    # ---- Markdown 报告 ----
    lines = []
    lines.append("# E1 实验统计报告：受控事实关系与回答行为分析\n")
    lines.append(f"- 模型：{data['model_path']}（bfloat16，贪心解码 do_sample=False）")
    lines.append(f"- 样本：{n_total} 条 = 30 组 × 4 条件（A1/A2/B1/B2）")
    lines.append(f"- 生成耗时：{data.get('elapsed_seconds', '?')}s，峰值显存 {data.get('max_peak_gb', '?')} GB")
    lines.append(f"- 报告时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    lines.append("## 1. 总体准确率\n")
    lines.append(f"- 可自动判定：{n_scored}/{n_total}，其中正确 **{ok}**，准确率 **{fmt_pct(overall)}**")
    lines.append(f"- 无法自动识别：{len(unclear)} 条（不计入准确率，原文见第 6 节）\n")

    lines.append("## 2. 四条件准确率\n")
    lines.append("| 条件 | 含义 | n | 可判定 | 正确 | 准确率 | 首句机构选择率 |")
    lines.append("|---|---|---|---|---|---|---|")
    zh = {"A1": "A正确·前半句", "A2": "A正确·后半句", "B1": "B正确·前半句", "B2": "B正确·后半句"}
    for c in CONDITIONS:
        s = by_cond[c]
        fp = f"{fmt_pct(s['first_half_pick_rate'])}" if s["first_half_pick_rate"] is not None else "-"
        lines.append(f"| {c} | {zh[c]} | {s['n']} | {s['scored']} | {s['correct']} | {fmt_pct(s['accuracy'])} | {fp} |")
    lines.append("")

    lines.append("## 3. 正确答案位置：前半句 vs 后半句\n")
    lines.append(f"- 前半句：{f_ok}/{f_n} = **{fmt_pct(f_acc)}**")
    lines.append(f"- 后半句：{s_ok}/{s_n} = **{fmt_pct(s_acc)}**")
    lines.append(f"- 两比例 z 检验：z = {z:.3f}，双侧 p = {p:.4f}\n")

    lines.append("## 4. 位置倾向\n")
    lines.append(f"- 首句机构（证据第一小句中的机构）被选率：**{fmt_pct(first_pick_rate)}**（n={len(all_picks)}；50% 表示无位置偏好，越高越倾向选第一个）")
    lines.append(f"- 候选 A 被选率：{fmt_pct(a_pick_rate)}（50% 表示无标签偏好）\n")

    lines.append("## 5. 分领域准确率\n")
    lines.append("| 领域 | n | 可判定 | 正确 | 准确率 |")
    lines.append("|---|---|---|---|---|")
    for d, s in by_domain.items():
        lines.append(f"| {d} | {s['n']} | {s['scored']} | {s['correct']} | {fmt_pct(s['accuracy'])} |")
    lines.append("")

    lines.append(f"## 6. 无法自动识别样本（{len(unclear)} 条，人工复核）\n")
    for r in unclear:
        lines.append(f"### {r['item_id']}（{r['eval']['status']} / {r['eval']['method']}）")
        lines.append(f"- 证据：{r['evidence']}")
        lines.append(f"- 金标：{r['gold_answer']}")
        lines.append(f"- 原始回答：`{r['answer_raw']}`\n")

    lines.append(f"## 7. 错误样本（自动判定为错误，共 {len(errors)} 条）\n")
    for r in errors:
        lines.append(f"### {r['item_id']}（金标 {r['gold_answer']}，模型答 {r['eval']['predicted_name']}）")
        lines.append(f"- 证据：{r['evidence']}")
        lines.append(f"- 原始回答：`{r['answer_raw']}`\n")

    out_md = out_dir / f"{args.out_prefix}_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # ---- 控制台摘要 ----
    print("=" * 70)
    print(f"总体：{ok}/{n_scored} 可判定正确 = {fmt_pct(overall)}（不确定 {len(unclear)} 条）")
    for c in CONDITIONS:
        s = by_cond[c]
        print(f"  {c}（{zh[c]}）: {s['correct']}/{s['scored']} = {fmt_pct(s['accuracy'])}"
              f"  首句选择率 {fmt_pct(s['first_half_pick_rate']) if s['first_half_pick_rate'] is not None else '-'}")
    print(f"前半句 {fmt_pct(f_acc)} vs 后半句 {fmt_pct(s_acc)}  (z={z:.3f}, p={p:.4f})")
    print(f"首句机构被选率 {fmt_pct(first_pick_rate)} | 候选A被选率 {fmt_pct(a_pick_rate)}")
    print(f"错误 {len(errors)} 条 | 无法自动识别 {len(unclear)} 条")
    print(f"已保存: {out_json.name}, {out_csv.name}, {out_md.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
