"""Two controlled evidence conditions; inspect every hidden-state layer."""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
EVIDENCE = {
    "2018": "检索证据：虚构的澜桥实验室成立于2018年。",
    "2021": "检索证据：虚构的澜桥实验室成立于2021年。",
}
QUESTION = "根据检索证据，澜桥实验室成立于哪一年？"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/hidden_states_minimal.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; confirm the project venv has a CUDA PyTorch build")
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=args.local_files_only,
    ).eval()

    states = {}
    cases = {}
    for label, evidence in EVIDENCE.items():
        messages = [{"role": "user", "content": evidence + "\n" + QUESTION}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True, use_cache=False)
        states[label] = [state[0, -1].float().cpu() for state in output.hidden_states]
        cases[label] = {
            "evidence": evidence,
            "question": QUESTION,
            "token_count": int(inputs.input_ids.shape[1]),
            "hidden_state_shapes": [list(state.shape) for state in output.hidden_states],
        }
        del output, inputs

    distances = [float(torch.linalg.vector_norm(a - b)) for a, b in zip(states["2018"], states["2021"])]
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": 42,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "dtype": "bfloat16",
        "state_index_0": "embedding output; subsequent indices are transformer block outputs",
        "measurement": "L2 distance between final prompt-token hidden vectors; descriptive only",
        "cases": cases,
        "per_state_l2_distance": distances,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
