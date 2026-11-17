"""E2：证据状态与回答行为分析 —— 行为判定与统计报告。

核心原则：证据状态（condition：充分/不足/冲突）与模型行为（behavior）分开判定，
绝不混同。行为判定保守，无法自动识别的单独记录，不强判。

行为分类（behavior）：
  says_conflict   输出「证据冲突/矛盾」类表述
  says_uncertain  输出「无法确定」类表述
  names_gold      作答段仅含金标机构核心词
  names_other     作答段仅含另一机构核心词
  mixed           同时含冲突与不确定表述，或两机构核心词都在（转人工复核）
  unrecognized    无法解析（转人工复核）

各条件下的「恰当行为」：
  sufficient   → names_gold        （正确回答）
  insufficient → says_uncertain    （恰当拒答）
  conflicting  → says_conflict     （识别冲突）
其余行为组合均单列计数（如不足条件下的 names_* = 无依据确定回答）。

运行：
    & 'D:\\workspace\\resume\\qlora-text2sql\\.venv\\Scripts\\python.exe' `
        'D:\\workspace\\rag-internal-mechanisms\\experiments\\e2_analyze.py'
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES = PROJECT_ROOT / "results" / "e2_responses.json"
CONDITIONS = ["sufficient", "insufficient", "conflicting"]

CONFLICT_MARKERS = ["冲突", "矛盾"]
UNCERTAINTY_MARKERS = [
    "无法确定", "不能确定", "无法判断", "无从判断", "不知道", "无法回答",
    "不能回答", "不足以", "信息不足", "资料不足", "没有提到", "未提到",
    "没有说明", "未说明", "无法得出", "尚不清楚",
]
NEGATION_MARKERS = ["不是", "而不是", "并非", "而非", "并不"]

APPROPRIATE = {"sufficient": "names_gold", "insufficient": "says_uncertain", "conflicting": "says_conflict"}
APPROPRIATE_ZH = {"sufficient": "给出金标机构", "insufficient": "回答无法确定", "conflicting": "回答证据冲突"}


def answer_span(raw: str) -> str:
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


def classify(item: dict, raw: str) -> dict:
    span = answer_span(raw)
    g_hit, o_hit = item["core_gold"] in span, item["core_other"] in span
    conflict = any(m in span for m in CONFLICT_MARKERS)
    uncertain = any(m in span for m in UNCERTAINTY_MARKERS)
    negation = any(m in span for m in NEGATION_MARKERS)
    if negation:
        behavior, note = "unrecognized", "作答段含否定表述，转人工复核"
    elif conflict and uncertain:
        behavior, note = "mixed", "同时含冲突与不确定表述"
    elif conflict:
        behavior, note = "says_conflict", "命中冲突表述"
    elif uncertain:
        behavior, note = "says_uncertain", "命中不确定性表述"
    elif g_hit and not o_hit:
        behavior, note = "names_gold", "仅含金标机构核心词"
    elif o_hit and not g_hit:
        behavior, note = "names_other", "仅含另一机构核心词"
    elif g_hit and o_hit:
        behavior, note = "mixed", "两机构核心词均在作答段"
    else:
        behavior, note = "unrecognized", "无标记无核心词"
    appropriate = (behavior == APPROPRIATE[item["condition"]])
    return {
        "answer_span": span,
        "behavior": behavior,
        "behavior_note": note,
        "appropriate": appropriate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 统计分析")
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--out-prefix", type=str, default="e2")
    args = parser.parse_args()

    data = json.loads(args.responses.read_text(encoding="utf-8"))
    rows = []
    for g in data["generations"]:
        ev = classify(g, g["answer_raw"])
        rows.append({**{k: g[k] for k in [
            "item_id", "group_id", "domain", "condition", "condition_zh", "question",
            "evidence", "gold_answer", "core_gold", "other_answer", "core_other",
            "answer_raw"]}, "eval": ev})

    n_total = len(rows)
    behaviors = ["names_gold", "names_other", "says_conflict", "says_uncertain", "mixed", "unrecognized"]

    by_cond = {}
    for c in CONDITIONS:
        sub = [r for r in rows if r["condition"] == c]
        counts = {b: sum(1 for r in sub if r["eval"]["behavior"] == b) for b in behaviors}
        n_ok = sum(1 for r in sub if r["eval"]["appropriate"])
        n_auto = len(sub) - counts["mixed"] - counts["unrecognized"]
        by_cond[c] = {
            "n": len(sub), "counts": counts,
            "appropriate": n_ok, "appropriate_rate": n_ok / len(sub) if sub else float("nan"),
            "scored": n_auto,
        }

    unclear = [r for r in rows if r["eval"]["behavior"] in ("mixed", "unrecognized")]

    # ---- 保存 ----
    out_dir = PROJECT_ROOT / "results"
    record = {
        "experiment": "e2_evidence_state_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiments/e2_analyze.py",
        "responses_file": str(args.responses),
        "n_total": n_total,
        "by_condition": by_cond,
        "rows": rows,
    }
    out_json = out_dir / f"{args.out_prefix}_scored.json"
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    flat = [{**{k: v for k, v in r.items() if k != "eval"}, **r["eval"]} for r in rows]
    out_csv = out_dir / f"{args.out_prefix}_rows.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)

    # ---- 报告 ----
    lines = []
    lines.append("# E2 实验统计报告：证据状态与回答行为分析\n")
    lines.append(f"- 模型：{data['model_path']}（bfloat16，贪心 do_sample=False, repetition_penalty=1.0）")
    lines.append(f"- 样本：{n_total} 条 = 30 组 × 3 条件（充分/不足/冲突），无候选字母，开放式作答")
    lines.append(f"- 生成耗时：{data.get('elapsed_seconds', '?')}s，峰值显存 {data.get('max_peak_gb', '?')} GB")
    lines.append(f"- 报告时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    lines.append("## 1. 条件 × 行为交叉表\n")
    lines.append("| 条件 | n | 给金标 | 给另一机构 | 答“证据冲突” | 答“无法确定” | 混合 | 无法解析 | 恰当行为率 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    zh_b = {"names_gold": "给金标", "names_other": "给另一机构", "says_conflict": "冲突",
            "says_uncertain": "不确定", "mixed": "混合", "unrecognized": "无法解析"}
    cond_zh = {"sufficient": "充分", "insufficient": "不足", "conflicting": "冲突"}
    for c in CONDITIONS:
        s = by_cond[c]
        cc = s["counts"]
        lines.append(
            f"| {c}（{cond_zh[c]}） "
            f"| {s['n']} | {cc['names_gold']} | {cc['names_other']} | {cc['says_conflict']} "
            f"| {cc['says_uncertain']} | {cc['mixed']} | {cc['unrecognized']} "
            f"| **{s['appropriate']}/{s['n']} = {s['appropriate_rate'] * 100:.1f}%** |")
    lines.append("")
    lines.append(f"恰当行为定义：充分→给出金标机构；不足→回答“无法确定”；冲突→回答“证据冲突”。\n")

    lines.append("## 2. 关键行为指标\n")
    ins = by_cond["insufficient"]["counts"]
    n_ins = by_cond["insufficient"]["n"]
    unfounded = ins["names_gold"] + ins["names_other"]
    lines.append(f"- **不足条件下的无依据确定回答：{unfounded}/{n_ins} = {unfounded / n_ins * 100:.1f}%**（编造/猜测具体机构，RAG 幻觉核心指标）")
    conf = by_cond["conflicting"]["counts"]
    n_conf = by_cond["conflicting"]["n"]
    missed = conf["names_gold"] + conf["names_other"]
    lines.append(f"- **冲突条件下的冲突漏检：{missed}/{n_conf} = {missed / n_conf * 100:.1f}%**（选边站而非识别冲突；其中选金标 {conf['names_gold']}、选另一机构 {conf['names_other']}）")
    suff = by_cond["sufficient"]["counts"]
    lines.append(f"- 充分条件：正确 {suff['names_gold']}/30，误答另一机构 {suff['names_other']}，误报冲突 {suff['says_conflict']}，误报不确定 {suff['says_uncertain']}")
    lines.append("")

    # 典型案例：条件与行为不匹配的全部列出
    lines.append("## 3. 值得注意的案例（行为与证据状态不匹配）\n")
    notable = {
        "sufficient": ["names_other", "says_conflict", "says_uncertain"],
        "insufficient": ["names_gold", "names_other", "says_conflict"],
        "conflicting": ["names_gold", "names_other", "says_uncertain"],
    }
    zh_pair = {
        ("sufficient", "names_other"): "充分却答另一机构",
        ("sufficient", "says_conflict"): "充分却误报冲突",
        ("sufficient", "says_uncertain"): "充分却拒答",
        ("insufficient", "names_gold"): "不足却给出金标（猜对）",
        ("insufficient", "names_other"): "不足却给出另一机构（编造）",
        ("insufficient", "says_conflict"): "不足却误报冲突",
        ("conflicting", "names_gold"): "冲突却选金标（漏检）",
        ("conflicting", "names_other"): "冲突却选另一机构（漏检）",
        ("conflicting", "says_uncertain"): "冲突却答无法确定（漏检）",
    }
    for c in CONDITIONS:
        for b in notable[c]:
            sub = [r for r in rows if r["condition"] == c and r["eval"]["behavior"] == b]
            if not sub:
                continue
            lines.append(f"### {zh_pair[(c, b)]}（{len(sub)} 条）")
            for r in sub:
                lines.append(f"- **{r['item_id']}** 金标：{r['gold_answer']}")
                lines.append(f"  - 证据：{r['evidence']}")
                lines.append(f"  - 原始回答：`{r['answer_raw']}`")
            lines.append("")

    lines.append(f"## 4. 无法自动识别样本（{len(unclear)} 条，人工复核）\n")
    for r in unclear:
        lines.append(f"- **{r['item_id']}**（{r['eval']['behavior']}）：`{r['answer_raw']}`")

    out_md = out_dir / f"{args.out_prefix}_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # ---- 控制台摘要 ----
    print("=" * 72)
    for c in CONDITIONS:
        s = by_cond[c]
        cc = s["counts"]
        print(f"{c:<12} n={s['n']}: 金标{cc['names_gold']:>2} 另机构{cc['names_other']:>2} "
              f"冲突{cc['says_conflict']:>2} 不确定{cc['says_uncertain']:>2} "
              f"混合{cc['mixed']} 解析失败{cc['unrecognized']} | 恰当 {s['appropriate']}/{s['n']}")
    print(f"不足条件无依据确定回答: {unfounded}/{n_ins}")
    print(f"冲突条件漏检: {missed}/{n_conf}（选金标 {conf['names_gold']}/选另一 {conf['names_other']}）")
    print(f"无法自动识别: {len(unclear)} 条")
    print(f"已保存: {out_json.name}, {out_csv.name}, {out_md.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
