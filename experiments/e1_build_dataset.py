"""E1：受控事实关系与回答行为分析 —— 样本构建与运行前自动校验。

设计
----
- 30 组虚构人物×机构事实，每组生成 A1/A2/B1/B2 四种条件，共 120 条：
    A1：候选 A 正确，答案（main 关系小句）在前半句
    A2：候选 A 正确，答案在后半句
    B1：候选 B 正确，答案在前半句
    B2：候选 B 正确，答案在后半句
- 同组四个版本仅小句顺序不同：实体、关键词、问句、候选完全相同。
- 四种条件由程序拼接产生（模板 + 换序），标签错误在构造上不可能；
  仍做全套断言校验（占位符、关键词、子句长度、唯一性、平衡性）。
- 候选 A/B 的机构归属、候选行展示顺序（AB/BA）按固定种子随机，
  避免模型利用固定顺序。
- 输入：data/e1_groups_raw.json（问句与四个小句模板，来自生成工作流）。
- 输出：data/e1_samples_v1.json（120 条完整测试数据）。

运行（借用只读环境，与之前实验一致）：
    & 'D:\\workspace\\resume\\qlora-text2sql\\.venv\\Scripts\\python.exe' `
        'D:\\workspace\\rag-internal-mechanisms\\experiments\\e1_build_dataset.py'
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUPS = PROJECT_ROOT / "data" / "e1_groups_raw.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "e1_samples_v1.json"
SEED = 20261117

# ---------------------------------------------------------------------------
# 中央分配表（唯一权威来源）：人名 / 机构 / 关系语义跨组唯一。
#   core1/core2   机构名核心词（用于回答匹配，两两非子串，且不为本组人名子串）
#   main_kw/distr_kw 关系关键词（用于校验小句语义归属，组内互斥）
# ---------------------------------------------------------------------------
ASSIGNMENTS = [
    # 大学
    dict(group_id="G01", domain="大学", person="沈既明", inst1="澜庭大学", inst2="朔野大学",
         core1="澜庭", core2="朔野", main="获得博士学位", distr="从事博士后研究",
         main_kw="博士学位", distr_kw="博士后", q_hint="在哪所大学获得博士学位"),
    dict(group_id="G02", domain="大学", person="顾昭桐", inst1="昭衡大学", inst2="屿声大学",
         core1="昭衡", core2="屿声", main="攻读硕士学位", distr="担任兼职讲师",
         main_kw="硕士学位", distr_kw="兼职讲师", q_hint="在哪所大学攻读硕士学位"),
    dict(group_id="G03", domain="大学", person="陆望澜", inst1="岑川大学", inst2="沐棠大学",
         core1="岑川", core2="沐棠", main="完成本科学业", distr="进行访问研究",
         main_kw="本科", distr_kw="访问研究", q_hint="在哪所大学完成本科学业"),
    dict(group_id="G04", domain="大学", person="温叙白", inst1="汀兰大学", inst2="栖迟大学",
         core1="汀兰", core2="栖迟", main="长期任教", distr="短期讲学",
         main_kw="任教", distr_kw="讲学", q_hint="在哪所大学长期任教"),
    dict(group_id="G05", domain="大学", person="池砚秋", inst1="竹屿大学", inst2="雪泾大学",
         core1="竹屿", core2="雪泾", main="出任法学院院长", distr="受聘客座教授",
         main_kw="院长", distr_kw="客座", q_hint="在哪所大学出任法学院院长"),
    dict(group_id="G06", domain="大学", person="郁疏桐", inst1="泠然大学", inst2="淮声大学",
         core1="泠然", core2="淮声", main="获授名誉博士学位", distr="开展合作研究",
         main_kw="名誉博士", distr_kw="合作研究", q_hint="在哪所大学获授名誉博士学位"),
    dict(group_id="G07", domain="大学", person="黎觉非", inst1="鹤汀大学", inst2="叙川大学",
         core1="鹤汀", core2="叙川", main="设立冠名奖学金", distr="担任产业顾问",
         main_kw="奖学金", distr_kw="顾问", q_hint="在哪所大学设立冠名奖学金"),
    dict(group_id="G08", domain="大学", person="桑无咎", inst1="拾翠大学", inst2="遂初大学",
         core1="拾翠", core2="遂初", main="主持联合实验室", distr="讲授暑期课程",
         main_kw="联合实验室", distr_kw="暑期", q_hint="在哪所大学主持联合实验室"),
    dict(group_id="G09", domain="大学", person="应雪岑", inst1="岫云大学", inst2="兰皋大学",
         core1="岫云", core2="兰皋", main="担任图书馆特藏部主任", distr="参与校史研究",
         main_kw="特藏", distr_kw="校史", q_hint="在哪所大学担任图书馆特藏部主任"),
    dict(group_id="G10", domain="大学", person="阮青崖", inst1="采菱大学", inst2="鹿溪大学",
         core1="采菱", core2="鹿溪", main="攻读第二学位", distr="指导田径队训练",
         main_kw="第二学位", distr_kw="田径", q_hint="在哪所大学攻读第二学位"),
    dict(group_id="G11", domain="大学", person="骆云笙", inst1="灼华大学", inst2="悠然大学",
         core1="灼华", core2="悠然", main="担任辩论队主教练", distr="开设通识讲座",
         main_kw="辩论队", distr_kw="通识", q_hint="在哪所大学担任辩论队主教练"),
    dict(group_id="G12", domain="大学", person="闵鹤知", inst1="澍雨大学", inst2="曲澜大学",
         core1="澍雨", core2="曲澜", main="获评杰出校友", distr="受邀毕业典礼演讲",
         main_kw="杰出校友", distr_kw="毕业典礼", q_hint="被哪所大学评为杰出校友"),
    # 公司
    dict(group_id="G13", domain="公司", person="祝长野", inst1="云枢科技", inst2="星阑信息",
         core1="云枢", core2="星阑", main="创立公司", distr="担任外部顾问",
         main_kw="创立", distr_kw="顾问", q_hint="创立了哪家公司"),
    dict(group_id="G14", domain="公司", person="屠文茵", inst1="曜石控股", inst2="燧石智能",
         core1="曜石", core2="燧石", main="出任首任首席执行官", distr="担任首席科学家",
         main_kw="首席执行官", distr_kw="首席科学家", q_hint="在哪家公司出任首任首席执行官"),
    dict(group_id="G15", domain="公司", person="符听澜", inst1="银汐科技", inst2="苇航科技",
         core1="银汐", core2="苇航", main="主持发布旗舰产品", distr="参与早期内测",
         main_kw="旗舰", distr_kw="内测", q_hint="在哪家公司主持发布旗舰产品"),
    dict(group_id="G16", domain="公司", person="简临镜", inst1="叠峰半导体", inst2="雾集半导体",
         core1="叠峰", core2="雾集", main="领导核心技术研发", distr="提供天使投资",
         main_kw="研发", distr_kw="投资", q_hint="在哪家公司领导核心技术研发"),
    dict(group_id="G17", domain="公司", person="晏清和", inst1="雷蜓集团", inst2="昆梧集团",
         core1="雷蜓", core2="昆梧", main="担任独立董事", distr="长期供职",
         main_kw="独立董事", distr_kw="供职", q_hint="在哪家公司担任独立董事"),
    dict(group_id="G18", domain="公司", person="封牧之", inst1="澈鸣信息", inst2="岿然物流",
         core1="澈鸣", core2="岿然", main="担任区域总经理", distr="完成管理培训生轮岗",
         main_kw="区域总经理", distr_kw="轮岗", q_hint="在哪家公司担任区域总经理"),
    dict(group_id="G19", domain="公司", person="竺星回", inst1="拓元传媒", inst2="千帆传媒",
         core1="拓元", core2="千帆", main="主导海外市场拓展", distr="承接短期本地化项目",
         main_kw="海外", distr_kw="本地化", q_hint="在哪家公司主导海外市场拓展"),
    dict(group_id="G20", domain="公司", person="濮怀瑾", inst1="步霆智能", inst2="屿安智能",
         core1="步霆", core2="屿安", main="发起企业开源项目", distr="签订商业授权合作",
         main_kw="开源", distr_kw="授权", q_hint="在哪家公司发起开源项目"),
    # 研究所
    dict(group_id="G21", domain="研究所", person="华嘉树", inst1="灵湫研究所", inst2="叠嶂研究所",
         core1="灵湫", core2="叠嶂", main="主持重大课题", distr="担任外聘评审",
         main_kw="课题", distr_kw="评审", q_hint="在哪个研究所主持重大课题"),
    dict(group_id="G22", domain="研究所", person="邵一苇", inst1="听松研究所", inst2="白鹭研究所",
         core1="听松", core2="白鹭", main="出任实验室主任", distr="参与联合培养项目",
         main_kw="实验室主任", distr_kw="联合培养", q_hint="在哪个研究所出任实验室主任"),
    dict(group_id="G23", domain="研究所", person="麦问渠", inst1="观星研究所", inst2="岱青研究所",
         core1="观星", core2="岱青", main="入选青年学者计划", distr="进行短期学术访问",
         main_kw="青年学者", distr_kw="学术访问", q_hint="入选了哪个研究所的青年学者计划"),
    dict(group_id="G24", domain="研究所", person="荀鹿鸣", inst1="沧溟研究所", inst2="烛霄研究所",
         core1="沧溟", core2="烛霄", main="牵头建设大科学装置", distr="承担数据处理协作",
         main_kw="大科学装置", distr_kw="数据处理", q_hint="在哪个研究所牵头建设大科学装置"),
    # 医院
    dict(group_id="G25", domain="医院", person="崔与归", inst1="青蕖医院", inst2="岚汀医院",
         core1="青蕖", core2="岚汀", main="完成住院医师规范化培训", distr="进修专项技术",
         main_kw="住院医师", distr_kw="进修", q_hint="在哪家医院完成住院医师规范化培训"),
    dict(group_id="G26", domain="医院", person="甘霁川", inst1="望岑医院", inst2="慕荷医院",
         core1="望岑", core2="慕荷", main="出任心内科主任", distr="担任特约会诊专家",
         main_kw="心内科", distr_kw="会诊", q_hint="在哪家医院出任心内科主任"),
    dict(group_id="G27", domain="医院", person="秋承屿", inst1="沁禾医院", inst2="青蘅医院",
         core1="沁禾", core2="青蘅", main="长期坐诊", distr="开展短期学术交流",
         main_kw="坐诊", distr_kw="学术交流", q_hint="在哪家医院长期坐诊"),
    # 文化机构
    dict(group_id="G28", domain="出版社", person="宁遂初", inst1="拾贝出版社", inst2="芹圃出版社",
         core1="拾贝", core2="芹圃", main="出版长篇处女作", distr="刊发中篇节选",
         main_kw="长篇", distr_kw="节选", q_hint="在哪家出版社出版了长篇处女作"),
    dict(group_id="G29", domain="乐团", person="柏彻", inst1="云璈乐团", inst2="鲸音爱乐乐团",
         core1="云璈", core2="鲸音", main="担任常任指挥", distr="担任客座指挥",
         main_kw="常任", distr_kw="客座", q_hint="在哪个乐团担任常任指挥"),
    dict(group_id="G30", domain="博物馆", person="白适野", inst1="九如博物馆", inst2="珮岚博物馆",
         core1="九如", core2="珮岚", main="策划年度特展", distr="提供展品借展支持",
         main_kw="特展", distr_kw="借展", q_hint="在哪个博物馆策划年度特展"),
]

CONDITIONS = ["A1", "A2", "B1", "B2"]
HALF_OF = {"A1": "first", "A2": "second", "B1": "first", "B2": "second"}
GOLD_IS_A = {"A1": True, "A2": True, "B1": False, "B2": False}
CLAUSE_SEP = "；"


def render(template: str, person: str, inst: str) -> str:
    return template.replace("{p}", person).replace("{i}", inst)


def collect_errors(assign: dict, gen: dict, issues: list) -> None:
    """把单个小组的所有校验错误追加到 issues（前缀 group_id）。"""
    gid = assign["group_id"]
    p, i1, i2 = assign["person"], assign["inst1"], assign["inst2"]
    c1, c2 = assign["core1"], assign["core2"]

    def err(msg: str) -> None:
        issues.append(f"[{gid}] {msg}")

    # 0) 生成结果与分配表逐字一致
    for key, val in [("person", p), ("institution_1", i1), ("institution_2", i2),
                     ("main_relation", assign["main"]), ("distractor_relation", assign["distr"])]:
        if gen.get(key) != val:
            err(f"{key} 与分配表不一致: {gen.get(key)!r} != {val!r}")

    q = gen.get("question", "")
    tpls = {k: gen.get(k, "") for k in
            ["main_first", "main_second", "distractor_first", "distractor_second"]}

    # 1) 占位符与标点规则
    for name in ["main_first", "distractor_first"]:
        t = tpls[name]
        if not t.startswith("{p}"):
            err(f"{name} 必须以 {{p}} 开头: {t!r}")
        if t.count("{p}") != 1 or t.count("{i}") != 1:
            err(f"{name} 中 {{p}}/{{i}} 必须各出现一次: {t!r}")
    for name in ["main_second", "distractor_second"]:
        t = tpls[name]
        if "{p}" in t:
            err(f"{name} 不得包含 {{p}}: {t!r}")
        if t.count("{i}") != 1:
            err(f"{name} 中 {{i}} 必须恰好出现一次: {t!r}")
    for name, t in tpls.items():
        if any(ch in t for ch in "；。，？！；,.;?!"):
            err(f"{name} 含句内标点: {t!r}")

    # 2) 问句规则
    if not q.startswith(p):
        err(f"问句必须以人名开头: {q!r}")
    if not q.endswith("？"):
        err(f"问句必须以「？」结尾: {q!r}")
    if c1 in q or c2 in q or i1 in q or i2 in q:
        err(f"问句泄漏机构名: {q!r}")

    # 3) 渲染后小句规则（主语 / 机构 / 关键词归属）
    rendered = {}
    for name in tpls:
        rendered[(name, 1)] = render(tpls[name], p, i1)
        rendered[(name, 2)] = render(tpls[name], p, i2)
    for name in ["main_first", "distractor_first"]:
        for k in (1, 2):
            if p not in rendered[(name, k)]:
                err(f"{name}(inst{k}) 渲染后缺人名: {rendered[(name, k)]!r}")
    for name in ["main_second", "distractor_second"]:
        for k in (1, 2):
            if p in rendered[(name, k)]:
                err(f"{name}(inst{k}) 渲染后不得再出现人名: {rendered[(name, k)]!r}")
    for name in ["main_first", "main_second", "distractor_first", "distractor_second"]:
        for k, other_core in ((1, c2), (2, c1)):
            r = rendered[(name, k)]
            inst_core = c1 if k == 1 else c2
            if inst_core not in r:
                err(f"{name}(inst{k}) 渲染后缺本机构核心词 {inst_core}: {r!r}")
            if other_core in r:
                err(f"{name}(inst{k}) 渲染后混入另一机构核心词 {other_core}: {r!r}")
    for k in (1, 2):
        m1, m2 = rendered[("main_first", k)], rendered[("main_second", k)]
        d1, d2 = rendered[("distractor_first", k)], rendered[("distractor_second", k)]
        for tag, clause in [("main_first", m1), ("main_second", m2)]:
            if assign["main_kw"] not in clause:
                err(f"{tag}(inst{k}) 缺 main 关键词「{assign['main_kw']}」: {clause!r}")
            if assign["distr_kw"] in clause:
                err(f"{tag}(inst{k}) 不应出现 distr 关键词「{assign['distr_kw']}」: {clause!r}")
        for tag, clause in [("distractor_first", d1), ("distractor_second", d2)]:
            if assign["distr_kw"] not in clause:
                err(f"{tag}(inst{k}) 缺 distr 关键词「{assign['distr_kw']}」: {clause!r}")
            if assign["main_kw"] in clause:
                err(f"{tag}(inst{k}) 不应出现 main 关键词「{assign['main_kw']}」: {clause!r}")

    # 4) 同条件两小句长度接近（控制非顺序差异）
    for k in (1, 2):
        for a_name, b_name in [("main_first", "distractor_first"), ("main_second", "distractor_second")]:
            la, lb = len(rendered[(a_name, k)]), len(rendered[(b_name, k)])
            if abs(la - lb) > 6:
                err(f"{a_name} vs {b_name} (inst{k}) 渲染长度差 {abs(la - lb)} > 6: {la} vs {lb}")


def build_items(assign: dict, gen: dict, rng: random.Random) -> tuple[list, dict]:
    """按固定种子随机分配候选 A/B 与候选行顺序，然后程序化拼接四种条件。"""
    a_is_inst1 = rng.random() < 0.5
    cand_order = "AB" if rng.random() < 0.5 else "BA"
    p = assign["person"]
    if a_is_inst1:
        cand_a, cand_b, core_a, core_b = assign["inst1"], assign["inst2"], assign["core1"], assign["core2"]
        main_t_a = gen["main_first"].replace("{i}", assign["inst1"])
        main_s_a = gen["main_second"].replace("{i}", assign["inst1"])
        dist_t_a = gen["distractor_first"].replace("{i}", assign["inst1"])
        dist_s_a = gen["distractor_second"].replace("{i}", assign["inst1"])
        main_t_b = gen["main_first"].replace("{i}", assign["inst2"])
        main_s_b = gen["main_second"].replace("{i}", assign["inst2"])
        dist_t_b = gen["distractor_first"].replace("{i}", assign["inst2"])
        dist_s_b = gen["distractor_second"].replace("{i}", assign["inst2"])
    else:
        cand_a, cand_b, core_a, core_b = assign["inst2"], assign["inst1"], assign["core2"], assign["core1"]
        main_t_a = gen["main_first"].replace("{i}", assign["inst2"])
        main_s_a = gen["main_second"].replace("{i}", assign["inst2"])
        dist_t_a = gen["distractor_first"].replace("{i}", assign["inst2"])
        dist_s_a = gen["distractor_second"].replace("{i}", assign["inst2"])
        main_t_b = gen["main_first"].replace("{i}", assign["inst1"])
        main_s_b = gen["main_second"].replace("{i}", assign["inst1"])
        dist_t_b = gen["distractor_first"].replace("{i}", assign["inst1"])
        dist_s_b = gen["distractor_second"].replace("{i}", assign["inst1"])

    main_t_a = main_t_a.replace("{p}", p)
    main_s_a = main_s_a.replace("{p}", p)
    dist_t_a = dist_t_a.replace("{p}", p)
    dist_s_a = dist_s_a.replace("{p}", p)
    main_t_b = main_t_b.replace("{p}", p)
    main_s_b = main_s_b.replace("{p}", p)
    dist_t_b = dist_t_b.replace("{p}", p)
    dist_s_b = dist_s_b.replace("{p}", p)

    # 四种条件：主语版小句在前，去主语版小句在后；仅顺序不同
    condition_clauses = {
        "A1": (main_t_a, dist_s_b),      # A 正确，答案在前半句
        "A2": (dist_t_b, main_s_a),      # A 正确，答案在后半句
        "B1": (main_t_b, dist_s_a),      # B 正确，答案在前半句
        "B2": (dist_t_a, main_s_b),      # B 正确，答案在后半句
    }

    items = []
    for cond in CONDITIONS:
        first, second = condition_clauses[cond]
        evidence = first + CLAUSE_SEP + second + "。"
        items.append({
            "item_id": f"{assign['group_id']}_{cond}",
            "group_id": assign["group_id"],
            "domain": assign["domain"],
            "condition": cond,
            "question": gen["question"],
            "evidence": evidence,
            "clause_first": first,
            "clause_second": second,
            "cand_a": cand_a,
            "cand_b": cand_b,
            "core_a": core_a,
            "core_b": core_b,
            "gold_answer": cand_a if GOLD_IS_A[cond] else cand_b,
            "gold_is_a": GOLD_IS_A[cond],
            "gold_half": HALF_OF[cond],          # 正确答案所在半句
            "cand_order": cand_order,            # 候选行展示顺序（组内恒定）
            "a_is_inst1": a_is_inst1,            # 候选 A 是否为分配行 inst1（组内恒定）
        })
    meta = {
        "group_id": assign["group_id"], "domain": assign["domain"], "person": p,
        "question": gen["question"],
        "main_relation": assign["main"], "distractor_relation": assign["distr"],
        "cand_a": cand_a, "cand_b": cand_b, "a_is_inst1": a_is_inst1, "cand_order": cand_order,
        "templates": {k: gen[k] for k in
                      ["main_first", "main_second", "distractor_first", "distractor_second"]},
    }
    return items, meta


def validate_items(all_items: list, all_groups: list, issues: list) -> None:
    """跨组唯一性 / 平衡性 / 条件正确性（构造后复核，防拼接 bug）。"""
    if len(all_items) != 120:
        issues.append(f"[ Balance ] 条目数 {len(all_items)} != 120")
    persons = [g["person"] for g in all_groups]
    insts = [g["cand_a"] for g in all_groups] + [g["cand_b"] for g in all_groups]
    if len(set(persons)) != 30:
        issues.append(f"[ Unique ] 人名重复: {len(set(persons))}/30")
    if len(set(insts)) != 60:
        issues.append(f"[ Unique ] 机构名重复: {len(set(insts))}/60")
    group_cores = []
    for it in all_items:
        if it["item_id"].endswith("_A1"):
            group_cores.append((it["core_a"], it["core_b"]))
    flat = [c for pair in group_cores for c in pair]
    if len(set(flat)) != 60:
        issues.append(f"[ Unique ] 核心词重复: {len(set(flat))}/60")
    for a, b in group_cores:
        if a in b or b in a:
            issues.append(f"[ Unique ] 核心词互为子串: {a} / {b}")
    # 平衡性
    n_cond = {c: sum(1 for it in all_items if it["condition"] == c) for c in CONDITIONS}
    if set(n_cond.values()) != {30}:
        issues.append(f"[ Balance ] 条件数不平衡: {n_cond}")
    n_gold_a = sum(1 for it in all_items if it["gold_is_a"])
    n_first = sum(1 for it in all_items if it["gold_half"] == "first")
    if n_gold_a != 60:
        issues.append(f"[ Balance ] gold=A 共 {n_gold_a} != 60")
    if n_first != 60:
        issues.append(f"[ Balance ] 前半句 gold 共 {n_first} != 60")
    # 每条：gold 机构只出现在对应半句的对应小句（按关键词+核心词复核条件正确性）
    for it in all_items:
        gold_core = it["core_a"] if it["gold_is_a"] else it["core_b"]
        other_core = it["core_b"] if it["gold_is_a"] else it["core_a"]
        gold_half_clause = it["clause_first"] if it["gold_half"] == "first" else it["clause_second"]
        other_half_clause = it["clause_second"] if it["gold_half"] == "first" else it["clause_first"]
        if gold_core not in gold_half_clause:
            issues.append(f"[{it['item_id']}] 正确答案核心词不在{it['gold_half']}半句")
        if other_core not in other_half_clause:
            issues.append(f"[{it['item_id']}] 干扰项核心词不在另一半句")
        if it["gold_answer"] not in it["evidence"]:
            issues.append(f"[{it['item_id']}] gold_answer 不在 evidence 中")
        # 证据必须恰好两个小句、以分号分隔
        if it["evidence"].count(CLAUSE_SEP) != 1:
            issues.append(f"[{it['item_id']}] evidence 小句数异常")


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 样本构建与自动校验")
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    raw = json.loads(args.groups.read_text(encoding="utf-8"))
    gens = {}
    for batch in raw:
        for g in batch["groups"]:
            gens[g["group_id"]] = g
    missing = [a["group_id"] for a in ASSIGNMENTS if a["group_id"] not in gens]
    if missing:
        sys.exit(f"缺少小组: {missing}")

    issues: list[str] = []
    rng = random.Random(args.seed)
    all_items, all_group_meta = [], []
    for assign in ASSIGNMENTS:
        gen = gens[assign["group_id"]]
        collect_errors(assign, gen, issues)
        items, meta = build_items(assign, gen, rng)
        all_items.extend(items)
        all_group_meta.append(meta)
    validate_items(all_items, all_group_meta, issues)

    print(f"校验完成：{len(issues)} 个问题")
    for msg in issues:
        print("  -", msg)
    if issues:
        sys.exit("存在校验错误，未写出样本文件（请先修复 data/e1_groups_raw.json）")

    n_a1 = sum(1 for g in all_group_meta if g["a_is_inst1"])
    n_ab = sum(1 for g in all_group_meta if g["cand_order"] == "AB")
    record = {
        "experiment": "e1_position_bias_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiments/e1_build_dataset.py",
        "seed": args.seed,
        "n_items": len(all_items),
        "n_groups": len(all_group_meta),
        "randomization": {
            "a_is_inst1_groups": n_a1,
            "cand_order_AB_groups": n_ab,
            "note": "候选A归属与候选行顺序按组随机（组内四条件恒定），种子固定可复现",
        },
        "validation": "全部断言通过：占位符/关键词/长度/唯一性/平衡性/条件正确性",
        "groups": all_group_meta,
        "items": all_items,
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {len(all_items)} 条样本 → {args.output}")
    print(f"  a_is_inst1: {n_a1}/30 组 | cand_order AB: {n_ab}/30 组")
    print("示例 A1:", all_items[0]["evidence"])


if __name__ == "__main__":
    main()
