"""E2.1：冲突数据集质量修正 —— 组装、程序化校验与逐条修改报告。

- 逐条对照 v1（data/e2_samples_v1.json 的 conflicting 条目 + 审计结论）与 v2 新第二小句。
- 程序化校验：实体一致、clause1 未变、机构用量按模式、元语言禁令、顺序连接词禁令、
  长度、互斥标记检测。语义互斥性无法程序化证明的（定指单称模式）标记 manual_review=True。
- 输出 data/e2_conflict_v2.json 与 results/e21_conflict_revision_report.md。
- 不修改 E0/E1/E2 任何原始文件，不运行模型。

运行：
    & 'D:\\workspace\\resume\\qlora-text2sql\\.venv\\Scripts\\python.exe' `
        'D:\\workspace\\rag-internal-mechanisms\\experiments\\e21_build_conflict_v2.py'
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "e21_audit_input.json"
DEFAULT_CRAFTED = PROJECT_ROOT / "data" / "e21_crafted_v2.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "e2_conflict_v2.json"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "e21_conflict_revision_report.md"

META_LANGUAGE = ["记录", "档案", "显示", "登记", "记载", "资料", "名单", "报道"]
SEQ_CONNECTIVES = ["随后", "其后", "后来", "之后", "此后", "又"]
NEG_MARKERS = ["并非", "并不是", "并不", "不是", "并未", "未在", "不在", "没有", "而非", "无关", "而在", "无法成立", "实为", "实则", "实际", "反倒", "错"]
UNIQUE_MARKERS = ["仅", "唯一", "只", "首次", "首度", "首位", "首任", "处女", "最早", "唯一一"]
DEFINITE_MARKERS = ["其", "该", "这场", "同一场", "这一", "此"]
SAME_EVENT_MARKERS = ["同一", "这场", "该", "当届", "当次", "此届", "这次", "这项"]

PATTERN_ZH = {
    "keep_v1": "保留v1（已是严格唯一性断言）",
    "neg_redirect": "直接否定+更正",
    "definite_singular": "定指单称",
    "uniqueness": "显式唯一性",
    "same_event": "同一事件",
}


def check_item(row: dict, crafted: dict, issues: list) -> dict:
    gid = row["group_id"]
    c1, c2 = row["clause1"], crafted["clause2"].strip()
    pattern = crafted.get("pattern_used", "")

    def err(msg):
        issues.append(f"[{gid}] {msg}")

    if c2.endswith("。") or c2.endswith("；"):
        err("clause2 结尾不应带标点")
    if "；" in c2:
        err("clause2 不应含「；」")
    if row["person"] in c2:
        err("clause2 不应含人名")
    for w in META_LANGUAGE:
        if w in c2:
            err(f"clause2 含元语言表述「{w}」")
    for w in SEQ_CONNECTIVES:
        if w in c2:
            err(f"clause2 含顺序连接词「{w}」")
    if abs(len(c2) - len(c1)) > 8:
        err(f"clause2 与 clause1 长度差 {abs(len(c2) - len(c1))} > 8")

    n_other = c2.count(row["core_other"])
    n_gold = c2.count(row["core_gold"])
    if pattern == "neg_redirect":
        if n_other != 1 or n_gold != 1:
            err(f"neg_redirect 应恰含两机构各一次（gold={n_gold}, other={n_other}）")
        if not any(m in c2 for m in NEG_MARKERS):
            err("neg_redirect 缺否定标记")
    else:
        if n_other != 1:
            err(f"应恰含另一机构一次（实际 {n_other} 次）")
        if n_gold != 0:
            err("不应含金标机构")
    if pattern == "uniqueness" and not any(m in c2 for m in UNIQUE_MARKERS):
        err("uniqueness 缺唯一性标记")
    if pattern == "definite_singular" and not any(m in c2 for m in DEFINITE_MARKERS):
        err("definite_singular 缺定指标记")
    if pattern == "same_event" and not any(m in c2 for m in SAME_EVENT_MARKERS):
        err("same_event 缺同一事件标记")
    if pattern == "keep_v1" and c2 != row["clause2_v1"]:
        err("keep_v1 模式 clause2 与 v1 不一致")

    manual_review = pattern == "definite_singular"  # 依赖问句单数预设，语义互斥无法程序化证明
    # 更严的通用规则：凡无强词汇互斥手段（否定/穷尽/唯一性计数）的句子，
    # 其互斥依赖同指、单数预设等语义判断，一律标记待人工复核
    STRONG = ["并非", "并不", "不是", "没有", "并未", "只有", "唯一", "仅", "别无", "一所", "首次", "最早", "首位", "首任", "处女"]
    if any(m in c2 for m in STRONG):
        manual_review = False
    elif pattern not in ("neg_redirect", "uniqueness", "keep_v1"):
        manual_review = True
    return {
        "exclusivity_markers": [m for m in NEG_MARKERS + UNIQUE_MARKERS + DEFINITE_MARKERS + SAME_EVENT_MARKERS if m in c2],
        "manual_review": manual_review,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E2.1 组装冲突 v2 并校验")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--crafted", type=Path, default=DEFAULT_CRAFTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    raw = json.loads(args.crafted.read_text(encoding="utf-8"))
    crafted, audits = {}, {}
    for batch in raw:
        for a in batch.get("audit", []):
            audits[a["group_id"]] = a
        for g in batch["groups"]:
            crafted[g["group_id"]] = g
    missing = [r["group_id"] for r in rows if r["group_id"] not in crafted]
    if missing:
        sys.exit(f"缺少重建产物: {missing}")
    missing_audit = [r["group_id"] for r in rows if r["group_id"] not in audits]
    if missing_audit:
        sys.exit(f"缺少审计结论: {missing_audit}")

    issues, items = [], []
    for row in rows:
        c = crafted[row["group_id"]]
        markers = check_item(row, c, issues)
        evidence = row["clause1"] + "；" + c["clause2"].strip() + "。"
        items.append({
            "group_id": row["group_id"],
            "domain": row["domain"],
            "question": row["question"],
            "person": row["person"],
            "gold": row["gold"],
            "other": row["other"],
            "core_gold": row["core_gold"],
            "core_other": row["core_other"],
            "main_relation": row["main_relation"],
            "clause1": row["clause1"],
            "clause2": c["clause2"].strip(),
            "evidence": evidence,
            "pattern": c.get("pattern_used", ""),
            "pattern_zh": PATTERN_ZH.get(c.get("pattern_used", ""), c.get("pattern_used", "")),
            "change_reason": c.get("change_reason", ""),
            "v1_clause2": row["clause2_v1"],
            "v1_evidence": row["clause1"] + "；" + row["clause2_v1"] + "。",
            "v1_audit_label": audits[row["group_id"]]["v1_label"],
            "v1_audit_reason": audits[row["group_id"]]["reason"],
            "v1_behavior": row.get("v1_behavior"),
            "v1_answer": row.get("v1_answer"),
            **markers,
        })

    # 汇总校验
    if len(items) != 30:
        issues.append(f"[Balance] 条目数 {len(items)} != 30")
    from collections import Counter
    pat_dist = dict(Counter(it["pattern"] for it in items))
    audit_dist = dict(Counter(it["v1_audit_label"] for it in items))
    n_manual = sum(1 for it in items if it["manual_review"])

    print(f"校验完成：{len(issues)} 个问题")
    for msg in issues:
        print("  -", msg)
    if issues:
        sys.exit("存在校验错误，未写出文件")
    print(f"v1 审计分布: {audit_dist}")
    print(f"v2 模式分布: {pat_dist}")
    print(f"待人工复核（定指单称模式）: {n_manual} 条")

    record = {
        "experiment": "e2_conflict_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiments/e21_build_conflict_v2.py",
        "note": "E2.1 仅修正冲突条件数据；不改动 E0/E1/E2 原始结果；未运行模型。",
        "v1_audit_distribution": audit_dist,
        "v2_pattern_distribution": pat_dist,
        "manual_review_count": n_manual,
        "validation": "程序化校验全部通过：实体/机构用量/元语言禁令/连接词/长度/互斥标记",
        "items": items,
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 逐条修改报告 ----
    lines = []
    lines.append("# E2.1 冲突数据集质量修正报告\n")
    lines.append(f"- 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- 范围：30 条冲突条件证据的第二小句重写（问句/人物/机构/第一小句不变）")
    lines.append(f"- v1 审计分布：{json.dumps(audit_dist, ensure_ascii=False)}")
    lines.append(f"- v2 互斥模式分布：{json.dumps(pat_dist, ensure_ascii=False)}")
    lines.append(f"- 待人工复核（定指单称，依赖问句单数预设）：{n_manual} 条")
    lines.append("- 判定标准：明确矛盾=构造不出自然同真解读；可能兼容=可自然构造同真解读；存在歧义=两种读法均自然\n")

    behavior_zh = {"says_conflict": "答“证据冲突”", "says_uncertain": "答“无法确定”",
                   "names_gold": "选金标机构", "names_other": "选另一机构"}
    for it in items:
        lines.append(f"## {it['group_id']}（{it['domain']}·{it['pattern_zh']}）\n")
        lines.append(f"- **问句**：{it['question']}")
        lines.append(f"- **v1 证据**：{it['v1_evidence']}")
        lines.append(f"  - 审计：**{it['v1_audit_label']}** —— {it['v1_audit_reason']}")
        if it["v1_behavior"]:
            lines.append(f"  - v1 时模型行为：{behavior_zh.get(it['v1_behavior'], it['v1_behavior'])}（`{it['v1_answer']}`）")
        lines.append(f"- **v2 证据**：{it['evidence']}")
        lines.append(f"  - 修改原因：{it['change_reason']}")
        marks = "、".join(it["exclusivity_markers"][:4]) or "（无显式标记）"
        lines.append(f"  - 互斥标记：{marks}" + ("　⚠️ **待人工复核**（语义互斥依赖问句单数预设）" if it["manual_review"] else ""))
        lines.append("")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写出 {len(items)} 条 → {args.output}")
    print(f"逐条修改报告 → {args.report}")


if __name__ == "__main__":
    main()
