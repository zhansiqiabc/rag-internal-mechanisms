"""E2：证据状态与回答行为分析 —— 用本地 Qwen2.5-3B-Instruct 生成回答。

- 本地加载，local_files_only=True；标准 Chat Template；贪心解码（do_sample=False，
  repetition_penalty=1.0 覆盖模型默认 1.05）；批量左填充。
- 统一 Prompt：不给候选、不用字母，模型可输出机构完整名称 / 「无法确定」 / 「证据冲突」。
- 证据状态标签（condition）只描述输入材料；模型行为由 e2_analyze.py 独立判定。

运行（借用只读环境）：
    & 'D:\\workspace\\resume\\qlora-text2sql\\.venv\\Scripts\\python.exe' `
        'D:\\workspace\\rag-internal-mechanisms\\experiments\\e2_run_inference.py' --limit 3 `
        --output 'D:\\workspace\\rag-internal-mechanisms\\results\\e2_smoke.json'
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hidden_states_sufficiency import (  # noqa: E402
    DEFAULT_MODEL,
    check_model_files,
    load_model_and_tokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = PROJECT_ROOT / "data" / "e2_samples_v1.json"

PROMPT_TEMPLATE = (
    "以下是一段人物资料和一个问题。请仅根据资料回答问题。\n\n"
    "资料：{evidence}\n"
    "问题：{question}\n\n"
    "回答规则：\n"
    "- 若资料能唯一确定答案，只输出该机构的完整名称；\n"
    "- 若资料不足以确定答案，只输出“无法确定”；\n"
    "- 若资料包含相互矛盾的信息，只输出“证据冲突”。\n"
    "不要输出任何其他内容。"
)


def build_user_content(item: dict) -> str:
    return PROMPT_TEMPLATE.format(evidence=item["evidence"], question=item["question"])


@torch.inference_mode()
def generate_batch(model, tokenizer, contents: list[str], max_new_tokens: int) -> list[dict]:
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True
        )
        for c in contents
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    n_prompt = enc.input_ids.shape[1]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        pad_token_id=pad_id,
    )
    eos_ids = {tokenizer.eos_token_id, pad_id}
    results = []
    for row in out:
        seq = row[n_prompt:].tolist()
        real_len = len(seq)
        for idx, tok in enumerate(seq):
            if tok in eos_ids:
                real_len = idx
                break
        text = tokenizer.decode(seq[:real_len], skip_special_tokens=True).strip()
        results.append({
            "answer_raw": text,
            "new_tokens": real_len,
            "truncated": bool(real_len >= max_new_tokens),
        })
    del enc, out
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 推理：90 条回答")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "e2_responses.json")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（冒烟测试用 3）")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    torch.manual_seed(42)
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    items = samples["items"]
    if args.limit:
        items = items[: args.limit]

    print("[1/3] 检查本地模型文件 ...")
    check_model_files(args.model)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请用带 CUDA 版 PyTorch 的解释器运行")

    print("[2/3] 加载模型（bfloat16, local_files_only=True）...")
    tokenizer, model = load_model_and_tokenizer(args.model, torch.bfloat16)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[3/3] 生成 {len(items)} 条回答（batch={args.batch_size}, greedy）...")
    rows = []
    t0 = time.perf_counter()
    max_peak = 0.0
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        contents = [build_user_content(it) for it in batch]
        results = generate_batch(model, tokenizer, contents, args.max_new_tokens)
        max_peak = max(max_peak, torch.cuda.max_memory_allocated() / 1024**3)
        for it, content, res in zip(batch, contents, results):
            rows.append({
                "item_id": it["item_id"],
                "group_id": it["group_id"],
                "domain": it["domain"],
                "condition": it["condition"],
                "condition_zh": it["condition_zh"],
                "question": it["question"],
                "evidence": it["evidence"],
                "gold_answer": it["gold_answer"],
                "core_gold": it["core_gold"],
                "other_answer": it["other_answer"],
                "core_other": it["core_other"],
                "user_content": content,
                "answer_raw": res["answer_raw"],
                "new_tokens": res["new_tokens"],
                "truncated": res["truncated"],
            })
            print(f"  [{len(rows):3d}/{len(items)}] {it['item_id']:<12} → {res['answer_raw'][:40]}")
    elapsed = time.perf_counter() - t0

    record = {
        "experiment": "e2_evidence_state_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiments/e2_run_inference.py",
        "samples_file": str(args.samples),
        "model_path": str(args.model),
        "decoding": "greedy (do_sample=False, repetition_penalty=1.0)",
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(elapsed, 1),
        "max_peak_gb": round(max_peak, 3),
        "n_items": len(rows),
        "generations": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{len(rows)} 条，耗时 {elapsed:.1f}s，峰值显存 {max_peak:.2f} GB")
    print(f"已保存 → {args.output}")


if __name__ == "__main__":
    main()
