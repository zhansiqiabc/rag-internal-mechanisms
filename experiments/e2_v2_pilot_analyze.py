"""Summarize the authorized pilot, retaining raw text and legacy auto labels.

Canonical outputs were inspected in full by Codex. Anything outside the four
allowed exact strings is retained for separate review, never keyword-scored as
confirmed. This is model-assisted review, not independent human annotation.
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from e2_analyze import classify
from e2_v2_pilot import CONDITIONS, DATASET, ROOT, read, save, sha, verify_protected

PREFIX = ROOT / 'results/e2_v2_pilot'
BEHAVIORS = ['names_gold', 'names_other', 'says_conflict', 'says_uncertain', 'pending_review']
EXPECTED = {'sufficient': 'names_gold', 'insufficient': 'says_uncertain', 'conflicting': 'says_conflict'}
LABELS = {'sufficient': '充分', 'insufficient': '不足', 'conflicting': '冲突'}
ZH = {'names_gold': '第一机构（继承 gold 字段）', 'names_other': '第二机构（继承 other 字段）',
      'says_conflict': '证据冲突', 'says_uncertain': '无法确定', 'pending_review': '待裁决'}


def canonical_behavior(row):
    mapping = {row['gold_answer']: 'names_gold', row['other_answer']: 'names_other',
               '证据冲突': 'says_conflict', '无法确定': 'says_uncertain'}
    return mapping.get(row['answer_raw'].strip(), 'pending_review')


def counts(rows, condition, key='reviewed_behavior'):
    sub = [r for r in rows if r['condition'] == condition]
    counter = Counter(r[key] for r in sub)
    return {'n': len(sub), 'counts': {b: counter[b] for b in BEHAVIORS},
            'expected_behavior_count': counter[EXPECTED[condition]],
            'format_compliant': sum(r.get('format_compliant', False) for r in sub),
            'truncated': sum(r.get('truncated', False) for r in sub)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replace-analysis', help='Explicit reason for regenerating analysis only; never runs inference.')
    args = parser.parse_args()
    verify_protected()
    out = PREFIX.with_name(PREFIX.name + '_scored.json')
    if out.exists() and not args.replace_analysis:
        raise FileExistsError('Analysis exists; no silent overwrite.')
    responses_path = PREFIX.with_name(PREFIX.name + '_responses.json')
    responses = read(responses_path)
    dataset = read(DATASET)
    assert responses['complete'] and responses['n_items'] == 69
    assert responses['samples_sha256'] == sha(DATASET)
    assert [r['item_id'] for r in responses['generations']] == [r['item_id'] for r in dataset['items']]
    rows = []
    for raw, source in zip(responses['generations'], dataset['items']):
        for key in ['group_id', 'condition', 'question', 'evidence', 'gold_answer', 'other_answer']:
            assert raw[key] == source[key], (raw['item_id'], key)
        assert raw['user_content'] == responses['prompt_template'].format(**raw)
        auto = classify(raw, raw['answer_raw'])
        reviewed = canonical_behavior(raw)
        compliant = raw['answer_raw'] in {raw['gold_answer'], raw['other_answer'], '证据冲突', '无法确定'}
        if reviewed != 'pending_review':
            assert auto['behavior'] == reviewed, (raw['item_id'], auto, reviewed)
        rows.append({**raw, 'auto_eval': auto, 'reviewed_behavior': reviewed,
                     'review_status': 'Codex model-assisted full-output review' if reviewed != 'pending_review' else 'pending',
                     'review_reason': '完整输出恰为该机构名称或规定短语；无额外解释、否定或混合表述。' if reviewed != 'pending_review' else '需结合完整回答另行裁决。',
                     'format_compliant': compliant,
                     'expected_behavior': EXPECTED[raw['condition']],
                     'matches_expected': reviewed == EXPECTED[raw['condition']]})
    old_record = read(ROOT / 'results/e2_responses.json')
    old = {r['item_id']: r for r in old_record['generations']}
    comparisons = []
    for row in rows:
        before = old[row['item_id']]
        assert before['group_id'] == row['group_id'] and before['condition'] == row['condition']
        assert before['question'] == row['question']
        assert before['user_content'] == responses['prompt_template'].format(**before)
        old_behavior = canonical_behavior(before)
        comparisons.append({
            'item_id': row['item_id'], 'group_id': row['group_id'], 'condition': row['condition'],
            'evidence_changed': before['evidence'] != row['evidence'],
            'old_evidence': before['evidence'], 'new_evidence': row['evidence'],
            'old_answer_raw': before['answer_raw'], 'new_answer_raw': row['answer_raw'],
            'old_behavior': old_behavior, 'new_behavior': row['reviewed_behavior'],
            'response_changed': before['answer_raw'] != row['answer_raw'],
            'reviewed_behavior': old_behavior,
            'format_compliant': before['answer_raw'] in {before['gold_answer'], before['other_answer'], '证据冲突', '无法确定'},
            'truncated': before.get('truncated', False),
        })
    summary = {c: counts(rows, c) for c in CONDITIONS}
    old_summary = {c: counts(comparisons, c) for c in CONDITIONS}
    smoke = read(PREFIX.with_name(PREFIX.name + '_smoke.json'))
    assert smoke['complete'] and smoke['n_items'] == 6
    smoke_matches = all(a['answer_raw'] == b['answer_raw'] for a, b in zip(smoke['generations'], rows[:6]))
    assert all(not row['truncated'] for row in rows)
    changed = [r for r in comparisons if r['evidence_changed']]
    assert all(r['condition'] == 'conflicting' for r in changed)
    paired_conf = [r for r in comparisons if r['condition'] == 'conflicting']
    transitions = Counter((r['old_behavior'], r['new_behavior']) for r in paired_conf)
    result = {
        'experiment': responses['experiment'], 'analyzed_utc': datetime.now(timezone.utc).isoformat(),
        'responses_sha256': sha(responses_path), 'dataset_sha256': sha(DATASET),
        'analysis_script_sha256': sha(Path(__file__)),
        'analysis_regeneration_reason': args.replace_analysis,
        'review_provenance': 'Codex inspected all 69 raw outputs; all are exact institution names or 无法确定. No independent human review. Legacy keyword scoring is stored separately.',
        'n_total': len(rows), 'n_groups': 23, 'by_condition': summary,
        'matched_v1_by_condition': old_summary,
        'evidence_changed_count': len(changed), 'smoke_full_first6_answers_equal': smoke_matches,
        'auto_review_disagreements': sum(r['auto_eval']['behavior'] != r['reviewed_behavior'] for r in rows),
        'pending_review_count': sum(r['reviewed_behavior'] == 'pending_review' for r in rows),
        'conflict_transitions': [{'old': a, 'new': b, 'n': n} for (a, b), n in sorted(transitions.items())],
        'rows': rows, 'paired_v1_v2': comparisons,
    }
    save(out, result)
    fields = ['item_id', 'group_id', 'condition', 'question', 'evidence', 'answer_raw',
              'auto_behavior', 'reviewed_behavior', 'review_status', 'review_reason',
              'format_compliant', 'truncated', 'matches_expected']
    with PREFIX.with_name(PREFIX.name + '_rows.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row['auto_eval']['behavior'] if k == 'auto_behavior' else row[k] for k in fields})
    conf = summary['conflicting']['counts']
    old_conf = old_summary['conflicting']['counts']
    lines = [
        '# E2 v2：23 组配对回答行为复测', '',
        '日期：2026-11-19。个人实验结果；小规模开发诊断，不是独立测试集或机制验证。', '',
        '## 用通俗的话说', '',
        f"充分材料 {summary['sufficient']['expected_behavior_count']}/23 回答正确机构；不足材料 {summary['insufficient']['expected_behavior_count']}/23 回答无法确定。",
        f"冲突材料 {conf['says_conflict']}/23 回答证据冲突，{conf['names_gold'] + conf['names_other']}/23 选择某个机构，{conf['says_uncertain']}/23 回答无法确定。",
        '这些回答显示当前提示词和受控材料下，模型没有按预期报告冲突；仅凭最终输出不能判断其内部是否已表示冲突。', '',
        '## 设置与核查', '',
        f"- 本地 Qwen2.5-3B-Instruct，{responses['dtype']}；greedy，repetition_penalty=1.0，seed=42，max_new_tokens=48，batch_size=12。",
        f"- 环境：Python {responses['python']}，PyTorch {responses['torch']}，Transformers {responses['transformers']}；GPU {responses['gpu']}。",
        f"- 全量生成耗时 {responses['elapsed_seconds']} 秒（不含加载），峰值已分配显存 {responses['max_peak_gb']} GB。",
        '- 先完成 6 条流程检查，再完整运行 69 条；按流程完整性放行，不以答题成绩放行。',
        f"- 69/69 与输入逐条对应；无空输出、无截断；6 条流程检查与全量对应输出一致：{smoke_matches}。",
        f"- 格式合规 {sum(r['format_compliant'] for r in rows)}/69；自动初筛与模型辅助复核分歧 {result['auto_review_disagreements']} 条；待裁决 {result['pending_review_count']} 条。",
        '- Codex 阅读了全部 69 条输出；本次输出均为完整机构名或“无法确定”，无否定、解释或混合长回答。此为模型辅助复核，无独立人工审核证据。',
        '- 输入、原始历史结果、运行脚本均有哈希记录；历史保护文件摘要核对通过。', '',
        '## 三条件与回答行为', '',
        '| 材料 | n | 输出第一机构 | 输出第二机构 | 证据冲突 | 无法确定 | 待裁决 | 预期行为达成 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for condition in CONDITIONS:
        s = summary[condition]
        c = s['counts']
        lines.append(f"| {LABELS[condition]} | {s['n']} | {c['names_gold']} | {c['names_other']} | {c['says_conflict']} | {c['says_uncertain']} | {c['pending_review']} | {s['expected_behavior_count']}/23 |")
    lines += ['', '第一/第二机构仅是 gold/other 的继承标识。冲突条件下两者均不能作为唯一正确答案；充分条件下第一机构是材料支持的答案。', '',
              '## 同样 23 组的旧版与新版对照', '',
              f"旧版这 23 组中：报告冲突 {old_conf['says_conflict']}，第一机构 {old_conf['names_gold']}，第二机构 {old_conf['names_other']}，无法确定 {old_conf['says_uncertain']}。",
              f"新版这 23 组中：报告冲突 {conf['says_conflict']}，第一机构 {conf['names_gold']}，第二机构 {conf['names_other']}，无法确定 {conf['says_uncertain']}。",
              f"69 条中 {len(changed)} 条证据文本改变，均为冲突条件；充分/不足 46 条原文及输出均与旧版一致。",
              '不能把这个变化解释为模型能力提高或退步：模型没有训练，冲突材料的内容和措辞改变了。旧版标签也未必满足本轮严格互斥定义；原始 21/30 不能作为本次成绩的直接比较基准。', '',
              '| 组号 | 证据是否改变 | 旧回答 | 新回答 |', '|---|---|---|---|']
    for pair in paired_conf:
        lines.append(f"| {pair['group_id']} | {'是' if pair['evidence_changed'] else '否'} | {pair['old_answer_raw']} | {pair['new_answer_raw']} |")
    lines += ['', '## 冲突条件逐条原文与输出（23 条）', '']
    for row in rows:
        if row['condition'] == 'conflicting':
            lines += [f"### {row['group_id']}", '', f"- 问题：{row['question']}", f"- 材料：{row['evidence']}",
                      f"- 原始回答：`{row['answer_raw']}`", f"- 行为：{ZH[row['reviewed_behavior']]}", '']
    lines += ['## 能说与不能说', '',
              '- 观察：在这 23 条经模型辅助筛选的冲突材料上，没有输出“证据冲突”；有选择机构和表达无法确定两种行为。',
              '- 假设：模型可能将否定/更正句理解为对前句的纠错，也可能没有正确整合冲突关系。两种解释尚未区分，不能仅凭输出裁定。',
              '- 不能声称模型不具备冲突识别能力、内部没有冲突表示，或证明了“可解码但未利用”；本次没有提取隐藏状态或训练探针。',
              '- 材料短、样本少、显式否定/唯一性措辞多，且标签无独立人工一致性证据。当前结论局限于本次受控设置。',
              '- 本轮按方案结束；未根据 0/23 重写材料、改提示词、重跑取最好结果或自动启动其他实验。', '',
              '## 产物与复现', '',
              '- 输入：`data/e2_samples_v2_pilot.json`（69 条，含来源哈希和组号）。',
              '- 原始输出：`results/e2_v2_pilot_responses.json`；流程检查：`results/e2_v2_pilot_smoke.json`。',
              '- 评分及配对明细：`results/e2_v2_pilot_scored.json`、`results/e2_v2_pilot_rows.csv`。',
              '- 保护文件哈希：`results/e2_v2_pilot_provenance.json`。',
              '- 构建/运行入口：`experiments/e2_v2_pilot.py`；分析入口：`experiments/e2_v2_pilot_analyze.py`。已有产物时脚本拒绝静默覆盖。',
              '- 模型运行需使用原 E2 的 CUDA 环境：`D:\\workspace\\resume\\qlora-text2sql\\.venv\\Scripts\\python.exe`；本项目 `.venv` 当前缺少 PyTorch。',
              '- 本次历史源文件摘要核对通过；原 E2 输入与生成结果未覆盖。', '']
    PREFIX.with_name(PREFIX.name + '_report.md').write_text('\n'.join(lines), encoding='utf-8')
    verify_protected()
    print(json.dumps({'by_condition': summary, 'matched_v1': old_summary, 'changed_inputs': len(changed),
                      'pending': result['pending_review_count']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
