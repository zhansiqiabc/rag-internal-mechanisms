"""证据充分性最小实验：提取"证据充分 / 证据不足"两种条件下的逐层隐藏状态。

设计说明
--------
- 只做前向传播与隐藏状态提取：不做微调、不做 Linear Probe、不做 Activation Patching。
- 两个样本共用同一个问题与同一个 Prompt 模板，仅证据句中一个短语不同
  （"获得博士学位" vs "从事博士后研究"），因此任何内部差异都只能来源于这个短语。
- 两条证据 token 数可能不同，完整的逐 token 隐藏矩阵不可直接对齐比较；
  但 chat template 末尾（generation prompt 结束处）的最后一个 token 在两个样本中
  是同一个 token（通常是 "\\n"），该位置的隐藏向量在两条件间语义对齐，可以比较。
- 逐层 L2 距离 / 余弦相似度只是描述性统计，不构成因果机制的证据。

运行方式（Windows PowerShell，项目根目录）
------------------------------------------
    .venv\\Scripts\\python.exe experiments\\hidden_states_sufficiency.py

模型从本地目录加载，local_files_only=True，不会联网、不会重新下载。
"""

import argparse
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 常量：项目路径、实验材料
# ---------------------------------------------------------------------------

# 以脚本位置定位项目根目录，保证从任意工作目录运行都正确
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "hidden_states_sufficiency.json"
DEFAULT_VECTOR_OUTPUT = PROJECT_ROOT / "results" / "hidden_states_sufficiency_lasttoken.pt"

QUESTION = "张三在哪所大学获得博士学位？"
PROMPT_TEMPLATE = "检索证据：{evidence}\n问题：{question}"

# 两个受控样本：仅证据句中的一个短语不同
SAMPLES = {
    "sufficient": {
        "label": "证据充分",
        "evidence": "张三在清华大学工作，后来在北京大学获得博士学位。",
    },
    "insufficient": {
        "label": "证据不足",
        "evidence": "张三在清华大学工作，后来在北京大学从事博士后研究。",
    },
}


# ---------------------------------------------------------------------------
# 基础检查与模型加载
# ---------------------------------------------------------------------------

def check_model_files(model_dir: Path) -> list[str]:
    """加载前检查本地模型文件是否齐全，缺权重时直接报错而不是尝试联网。"""
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"未找到 {model_dir / 'config.json'}，模型目录不正确")
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
        missing = [s for s in shards if not (model_dir / s).is_file()]
        if missing:
            raise FileNotFoundError(
                f"模型权重分片缺失: {missing}。模型未完整下载，本脚本不联网，请先补全权重。"
            )
        return shards
    # 单文件权重模型没有 index.json
    return []


def load_model_and_tokenizer(model_dir: Path, dtype: torch.dtype):
    """从本地路径加载模型与分词器（BF16 优先，不联网，eval 模式）。"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # transformers >=4.56 参数名为 dtype=，更早版本为 torch_dtype=，做兼容
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, device_map="cuda", local_files_only=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, device_map="cuda", local_files_only=True
        )
    model.eval()  # 关闭 dropout 等训练期行为，前向才可复现
    # 校验实际加载精度，防止旧版本把 dtype= 当作未知参数忽略后静默用 FP32
    actual_dtype = next(model.parameters()).dtype
    if actual_dtype != dtype:
        raise RuntimeError(
            f"模型实际加载为 {actual_dtype}，与请求的 {dtype} 不符"
            f"（transformers {transformers.__version__} 可能不支持 dtype= 参数名）。"
        )
    return tokenizer, model


# ---------------------------------------------------------------------------
# 前向提取
# ---------------------------------------------------------------------------

def extract_last_token_states(model, tokenizer, evidence: str):
    """对一个样本做一次前向，返回（最后 token 的逐层隐藏向量，元信息）。

    hidden_states 说明：
      - 返回 tuple，第 0 项是 embedding 层输出，其后每经过一个 Transformer 块追加一项；
      - 我们只取序列最后一个 token（位置 -1）在各层的向量，转成 float32 存 CPU，
        避免占用 GPU 显存，也便于后续保存与复用。
    """
    content = PROMPT_TEMPLATE.format(evidence=evidence, question=QUESTION)
    messages = [{"role": "user", "content": content}]
    # 套用模型自带 chat template，加 generation prompt（末尾为 assistant 起始符）
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    torch.cuda.reset_peak_memory_stats()  # 峰值显存只统计本次前向
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, use_cache=False)

    last_token_states = [h[0, -1].float().cpu() for h in output.hidden_states]
    meta = {
        "prompt": prompt,
        "token_count": int(inputs.input_ids.shape[1]),
        "last_token_str": tokenizer.decode(inputs.input_ids[0, -1]),
        "hidden_state_shapes": [list(h.shape) for h in output.hidden_states],
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }
    del output, inputs
    torch.cuda.empty_cache()
    return last_token_states, meta


def finite_layers(states: list[torch.Tensor]) -> list[int]:
    """返回出现 NaN 或 Inf 的层下标；空列表表示全部数值正常。"""
    return [i for i, s in enumerate(states) if not bool(torch.isfinite(s).all())]


def compare_states(states_a: list[torch.Tensor], states_b: list[torch.Tensor]) -> dict:
    """逐层计算最后 token 向量的 L2 距离与余弦相似度（均为描述性统计）。"""
    l2 = [float(torch.linalg.vector_norm(a - b)) for a, b in zip(states_a, states_b)]
    cos = [float(F.cosine_similarity(a, b, dim=0)) for a, b in zip(states_a, states_b)]
    return {"l2_distance": l2, "cosine_similarity": cos}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="证据充分性最小隐藏状态实验")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="本地模型目录")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16",
                        help="加载精度，默认 bfloat16（RTX 50 系原生支持）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON 结果输出路径")
    parser.add_argument("--vector-output", type=Path, default=DEFAULT_VECTOR_OUTPUT,
                        help="最后 token 逐层向量的 .pt 保存路径（供后续阶段复用）")
    args = parser.parse_args()

    # 固定随机种子（前向本身确定，但按规范固定，保证整体可复现）
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[1/5] 检查本地模型文件 ...")
    shards = check_model_files(args.model)
    print(f"      权重分片齐全: {shards}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请确认所用 venv 安装的是 CUDA 版 PyTorch，且驱动正常")
    dtype = getattr(torch, args.dtype)

    print(f"[2/5] 加载模型（{args.dtype}, local_files_only=True）...")
    tokenizer, model = load_model_and_tokenizer(args.model, dtype)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    mem_after_load = torch.cuda.memory_allocated() / 1024**3
    print(f"      Qwen2.5-3B-Instruct: {n_layers} 层, hidden_size={hidden_size}, "
          f"加载后显存 {mem_after_load:.2f} GB")

    # ---- 两个样本各做一次前向 ----
    states: dict[str, list[torch.Tensor]] = {}
    cases: dict[str, dict] = {}
    step = 3
    for key, sample in SAMPLES.items():
        print(f"[{step}/5] 样本 {key}（{sample['label']}）前向 ...")
        sample_states, meta = extract_last_token_states(model, tokenizer, sample["evidence"])
        states[key] = sample_states
        cases[key] = {
            "label": sample["label"],
            "evidence": sample["evidence"],
            "token_count": meta["token_count"],
            "last_token_str": meta["last_token_str"],
            "prompt": meta["prompt"],
            "hidden_state_shapes": meta["hidden_state_shapes"],
            "last_token_vector_norm_per_layer": [
                round(float(torch.linalg.vector_norm(s)), 6) for s in sample_states
            ],
            "peak_allocated_gb": meta["peak_allocated_gb"],
            "peak_reserved_gb": meta["peak_reserved_gb"],
            "nan_inf_layers": finite_layers(sample_states),
        }
        print(f"      tokens={meta['token_count']}, 最后 token={meta['last_token_str']!r}, "
              f"峰值显存 {meta['peak_allocated_gb']:.2f} GB")
        step += 1

    # ---- 逐层比较（描述性统计）----
    comparison = compare_states(states["sufficient"], states["insufficient"])
    l2 = comparison["l2_distance"]

    # ---- 可重复性检查：同一输入再跑一次，比较是否得到相同向量 ----
    print("[5/5] 可重复性检查（样本 sufficient 重跑一次）...")
    rerun_states, _ = extract_last_token_states(
        model, tokenizer, SAMPLES["sufficient"]["evidence"]
    )
    rerun_max_diff = max(
        float((a - b).abs().max()) for a, b in zip(states["sufficient"], rerun_states)
    )
    rerun_allclose = all(
        torch.allclose(a, b, rtol=1e-3, atol=1e-4)
        for a, b in zip(states["sufficient"], rerun_states)
    )

    n_states = len(states["sufficient"])
    # 隐藏状态结构（transformers 4.x 与 5.x 行为一致，已在 5.20.1 源码中核实）：
    #   总数 = 层数 + 1；第 0 项 = embedding 输出；第 1..N-1 项 = 对应块的原始输出；
    #   最后一项 = 最后一个块经过最终 RMSNorm 之后的输出（即送入 lm_head 前的向量，
    #   5.x 中由 tie_last_hidden_states=True 用 last_hidden_state 覆盖）。
    # 因此倒数第二项对应第 N-1 块，最后一个块的"原始"输出不出现在返回值里。
    last_is_final_norm = n_states == n_layers + 1
    if not last_is_final_norm:
        print(f"警告: 隐藏状态数量({n_states})与层数+1({n_layers + 1})不符，请核对 transformers 版本行为")

    # ---- 汇总验证结果 ----
    verification = {
        "all_finite_sufficient": len(cases["sufficient"]["nan_inf_layers"]) == 0,
        "all_finite_insufficient": len(cases["insufficient"]["nan_inf_layers"]) == 0,
        "last_vector_dim_equal": states["sufficient"][0].shape == states["insufficient"][0].shape,
        "layer_count_matches_config": n_states in (n_layers + 1, n_layers + 2),
        "state0_l2_distance_expected_zero": l2[0],
        "reproducibility": {
            "rerun_sample": "sufficient",
            "max_abs_diff": rerun_max_diff,
            "allclose_rtol1e-3": rerun_allclose,
        },
    }

    record = {
        "experiment": "sufficiency_hidden_states_minimal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiments/hidden_states_sufficiency.py",
        "model_path": str(args.model),
        "local_files_only": True,
        "seed": args.seed,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "gpu_total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "dtype": args.dtype,
        "design": {
            "question": QUESTION,
            "prompt_template": PROMPT_TEMPLATE,
            "samples": SAMPLES,
            "alignment_note": (
                "两条件 token 数可能不同，完整逐 token 隐藏矩阵不可直接比较；"
                "chat template 末尾最后一个 token 两条件相同，该位置向量可对齐比较。"
            ),
        },
        "model_structure": {
            "num_hidden_layers": n_layers,
            "hidden_size": hidden_size,
            "num_hidden_states_returned": n_states,
            "last_state_is_after_final_norm": last_is_final_norm,
            "state_index_0": "embedding 层输出；1..N-1 为各块原始输出；最后一项为最终 RMSNorm 后（lm_head 输入）的向量",
        },
        "cases": cases,
        "per_state_comparison": comparison,
        "verification": verification,
        "interpretation_caveat": (
            "层间 L2 距离与余弦相似度仅是描述性测量；未经探针训练或干预实验验证，"
            "不能解释为因果机制，也不能证明证据信息被用于生成答案。"
        ),
    }

    # ---- 保存 JSON 与最后 token 向量（.pt 供后续 Linear Probe 阶段复用）----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    args.vector_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sufficient": torch.stack(states["sufficient"]),        # [层数(+1), hidden]
            "insufficient": torch.stack(states["insufficient"]),
            "meta": {"num_layers": n_layers, "hidden_size": hidden_size,
                     "dtype": "float32", "seed": args.seed},
        },
        args.vector_output,
    )

    # ---- 控制台摘要 ----
    max_l2_idx = max(range(n_states), key=lambda i: l2[i])
    print()
    print("=" * 62)
    print(f"层数: {n_layers} | hidden_size: {hidden_size} | 状态数: {n_states}")
    print(f"embedding 层(state 0) L2 = {l2[0]:.6f}（相同 token，理论上为 0）")
    print(f"L2 最大值 {max(l2):.4f} @ state {max_l2_idx} | 末层 L2 = {l2[-1]:.4f}")
    print(f"NaN/Inf 层: 充分={cases['sufficient']['nan_inf_layers']} "
          f"不足={cases['insufficient']['nan_inf_layers']}")
    print(f"维度一致: {verification['last_vector_dim_equal']} | "
          f"重跑最大差异: {rerun_max_diff:.3e} | allclose: {rerun_allclose}")
    print(f"已保存: {args.output}")
    print(f"已保存: {args.vector_output}")
    print("=" * 62)


if __name__ == "__main__":
    main()
