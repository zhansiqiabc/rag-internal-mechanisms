"""E2：证据状态与回答行为分析 —— 样本构建与运行前自动校验。

设计
----
- 复用 E1 的 30 组实体（data/e1_groups_raw.json + e1_build_dataset.ASSIGNMENTS），
  每组三种证据条件，共 90 条：
    sufficient   充分：主关系句(金标机构)；干扰关系句(另一机构)。程序拼接，复用 E1 模板。
    insufficient 不足：两个干扰关系小句，两机构都出现但都不回答问题（由创作工作流
                 撰写/微调，避免定指模板造成隐性矛盾或泄漏答案）。
    conflicting  冲突：主关系句(金标机构，与充分条件第一小句逐字相同)；
                 第二小句就同一事实声称另一机构（创作工作流撰写，确保矛盾是唯一自然解读）。
- 三条件同问句、同实体、两小句结构、长度接近；金标机构归属按固定种子随机（18/12 分布）。
- 证据状态标签（condition）与模型行为分开记录，评估阶段不混同。

运行：
    & 'D:\\workspace\\resume\\qlora-text2sql\\.venv\\Scripts\\python.exe' `
        'D:\\workspace\\rag-internal-mechanisms\\experiments\\e2_build_dataset.py'
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "e2_craft_input.json"
DEFAULT_CRAFTED = PROJECT_ROOT / "data" / "e2_crafted.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "e2_samples_v1.json"
CLAUSE_SEP = "；"
CONDITIONS = ["sufficient", "insufficient", "conflicting"]
CONDITION_ZH = {"sufficient": "充分", "insufficient": "不足", "conflicting": "冲突"}
LENGTH_TOL = 10


def collect_errors(row: dict, crafted: dict, issues: list) -> None:
    gid = row["group_id"]
    p, G, O = row["person"], row["gold"], row["other"]
    cg, co = row["core_gold"], row["core_other"]
    insufficient = crafted["insufficient_evidence"]
    conflict_second = crafted["conflict_second"]

    def err(msg):
        issues.append(f"[{gid}] {msg}")

    # ---- 通用结构 ----
    for name, ev in [("insufficient_evidence", insufficient)]:
        if ev.count(CLAUSE_SEP) != 1:
            err(f"{name} 小句数异常: {ev!r}")
        if not ev.endswith("。"):
            err(f"{name} 应以「。」结尾: {ev!r}")

    # ---- 充分（程序拼接，校验防 bug）----
    sufficient = row["sufficient_evidence"]
    s1, s2 = sufficient[:-1].split(CLAUSE_SEP)
    if not s1.startswith(p):
        err(f"充分条件第一小句应以人名开头: {s1!r}")
    if p in s2:
        err(f"充分条件第二小句不应含人名: {s2!r}")
    if cg not in s1 or co not in s2:
        err(f"充分条件机构位置错误: {s1!r} / {s2!r}")
    if row["main_kw"] not in s1 or row["main_kw"] in s2:
        err(f"充分条件 main_kw 归属错误（应仅在第一小句）")
    if row["distr_kw"] not in s2 or row["distr_kw"] in s1:
        err(f"充分条件 distr_kw 归属错误（应仅在第二小句）")

    # ---- 不足 ----
    i1, i2 = insufficient[:-1].split(CLAUSE_SEP)
    if not i1.startswith(p):
        err(f"不足条件第一小句应以人名开头: {i1!r}")
    if p in i2:
        err(f"不足条件第二小句不应含人名: {i2!r}")
    if cg not in i1 or co not in i2:
        err(f"不足条件机构位置错误（gold 应在第一小句、other 在第二小句）: {i1!r} / {i2!r}")
    if cg in i2 or co in i1:
        err(f"不足条件机构跨小句混入: {i1!r} / {i2!r}")
    if row["main_kw"] in insufficient:
        err(f"不足条件泄漏 main_kw「{row['main_kw']}」: {insufficient!r}")
    if abs(len(insufficient) - len(sufficient)) > LENGTH_TOL:
        err(f"不足条件与充分条件长度差 {abs(len(insufficient) - len(sufficient))} > {LENGTH_TOL}")

    # ---- 冲突 ----
    if not conflict_second.strip():
        err("conflict_second 为空")
    c1 = row["clause1_main_G"]
    conflicting = c1 + CLAUSE_SEP + conflict_second.strip() + "。"
    if p in conflict_second:
        err(f"冲突第二小句不应含人名: {conflict_second!r}")
    if cg not in c1:
        err("冲突第一小句缺金标机构")
    if co not in conflict_second:
        err(f"冲突第二小句缺 other 机构: {conflict_second!r}")
    if cg in conflict_second:
        err(f"冲突第二小句混入金标机构: {conflict_second!r}")
    if row["main_kw"] not in conflict_second and row["main_kw"] not in c1:
        err("冲突条件 main_kw 缺失（两句应各表达 main 关系语义）")
    for w in ["随后", "其后", "后来", "之后"]:
        if w in conflict_second:
            err(f"冲突第二小句含顺序连接词「{w}」: {conflict_second!r}")
    if abs(len(conflicting) - len(sufficient)) > LENGTH_TOL:
        err(f"冲突条件与充分条件长度差 {abs(len(conflicting) - len(sufficient))} > {LENGTH_TOL}")
    # 问句不得泄漏机构
    if cg in row["question"] or co in row["question"]:
        err("问句泄漏机构名")


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 样本构建与自动校验")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--crafted", type=Path, default=DEFAULT_CRAFTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    crafted_raw = json.loads(args.crafted.read_text(encoding="utf-8"))
    crafted = {}
    for batch in crafted_raw:
        for g in batch["groups"]:
            crafted[g["group_id"]] = g
    missing = [r["group_id"] for r in rows if r["group_id"] not in crafted]
    if missing:
        sys.exit(f"缺少创作产物: {missing}")

    issues = []
    items = []
    for row in rows:
        c = crafted[row["group_id"]]
        collect_errors(row, c, issues)
        conflicting = row["clause1_main_G"] + CLAUSE_SEP + c["conflict_second"].strip() + "。"
        evidences = {
            "sufficient": row["sufficient_evidence"],
            "insufficient": c["insufficient_evidence"].strip(),
            "conflicting": conflicting,
        }
        for cond in CONDITIONS:
            items.append({
                "item_id": f"{row['group_id']}_{cond[:4].lower()}",
                "group_id": row["group_id"],
                "domain": row["domain"],
                "condition": cond,
                "condition_zh": CONDITION_ZH[cond],
                "question": row["question"],
                "evidence": evidences[cond],
                "gold_answer": row["gold"],
                "core_gold": row["core_gold"],
                "other_answer": row["other"],
                "core_other": row["core_other"],
                "gold_is_inst1": row["gold_is_inst1"],
                "main_relation": row["main_relation"],
                "distractor_relation": row["distractor_relation"],
            })

    # ---- 平衡性与唯一性 ----
    n_cond = {c: sum(1 for it in items if it["condition"] == c) for c in CONDITIONS}
    if set(n_cond.values()) != {30}:
        issues.append(f"[Balance] 条件数不平衡: {n_cond}")
    if len({it["question"] for it in items}) != 30:
        issues.append("[Unique] 问句数 != 30")
    per_group = {}
    for it in items:
        per_group.setdefault(it["group_id"], set()).add(it["condition"])
    if any(v != set(CONDITIONS) for v in per_group.values()):
        issues.append("[Balance] 存在小组条件不全")

    print(f"校验完成：{len(issues)} 个问题")
    for msg in issues:
        print("  -", msg)
    if issues:
        sys.exit("存在校验错误，未写出样本文件")

    record = {
        "experiment": "e2_evidence_state_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiments/e2_build_dataset.py",
        "n_items": len(items),
        "n_groups": len(rows),
        "gold_is_inst1_groups": sum(1 for r in rows if r["gold_is_inst1"]),
        "validation": "全部断言通过：结构/机构位置/关键词/长度/平衡性",
        "items": items,
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {len(items)} 条样本 → {args.output}")
    ex = items[0]
    print("示例 充分:", ex["evidence"])
    print("示例 不足:", items[1]["evidence"])
    print("示例 冲突:", items[2]["evidence"])


if __name__ == "__main__":
    main()
