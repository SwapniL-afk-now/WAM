"""Precompute umT5 text embeddings for every LIBERO(-Plus) instruction.

Mirrors ``FastWAM.encode_prompt`` exactly (tokenizer_max_len=128, padded
positions zeroed, all-ones mask) so the policy never loads the 5.7B encoder.

    python scripts/build_prompt_bank.py --out $IMAGO_CKPT_DIR/prompt_bank_liberoplus.pt
"""

from __future__ import annotations

import argparse
import os

import torch

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def collect_instructions(libero_type: str, suites) -> list[str]:
    if libero_type == "plus":
        from liberoplus.liberoplus import benchmark
    else:
        from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()
    instructions = []
    for suite_name in suites:
        suite = bench[suite_name]()
        for task_id in range(suite.n_tasks):
            instructions.append(str(suite.get_task(task_id).language))
    return sorted(set(instructions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--libero-type", default=os.environ.get("LIBERO_TYPE", "plus"))
    parser.add_argument("--suites", nargs="+", default=list(SUITES))
    parser.add_argument("--extra", default=None, help="txt file with extra instructions")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer-model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    tasks = collect_instructions(args.libero_type, args.suites)
    if args.extra:
        with open(args.extra) as f:
            tasks += [line.strip() for line in f if line.strip()]
        tasks = sorted(set(tasks))
    prompts = [DEFAULT_PROMPT.format(task=t) for t in tasks]
    print(f"[prompt_bank] {len(prompts)} unique prompts")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=True,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    encoder = _load_registered_model(
        text_config.path, "wan_video_text_encoder", torch_dtype=torch.bfloat16, device=device
    ).eval()
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=args.max_len, clean="whitespace")

    contexts, masks = [], []
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            ids, mask = tokenizer(batch, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device=device, dtype=torch.bool)
            emb = encoder(ids, mask)
            lens = mask.gt(0).sum(dim=1).long()
            for i, n in enumerate(lens):
                emb[i, n:] = 0  # same as FastWAM.encode_prompt
            contexts.append(emb.to("cpu", torch.bfloat16))
            masks.append(torch.ones_like(mask, device="cpu"))
    torch.save(
        {
            "prompts": prompts,
            "context": torch.cat(contexts),
            "context_mask": torch.cat(masks),
            "max_len": args.max_len,
            "tokenizer_model_id": args.tokenizer_model_id,
        },
        args.out,
    )
    print(f"[prompt_bank] saved {args.out}")


if __name__ == "__main__":
    main()
