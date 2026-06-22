#!/usr/bin/env python3
"""Run inference with a merged Hugging Face checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a ReTool-style response from a merged HF model."
    )
    parser.add_argument("--model", required=True, help="HF model directory or model id.")
    parser.add_argument(
        "--tokenizer",
        help="Optional tokenizer directory. Defaults to --model.",
    )
    parser.add_argument("--question", required=True, help="Problem text to solve.")
    parser.add_argument(
        "--prompt-template",
        default="prompts/solve_with_code.txt",
        help="Prompt template path containing {question}.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="Model dtype. auto uses bf16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map, for example auto, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Feed the raw prompt instead of tokenizer.apply_chat_template.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def first_model_device(model: torch.nn.Module) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            text = str(device)
            if text.startswith("cuda") or text.isdigit():
                return torch.device(f"cuda:{text}" if text.isdigit() else text)
    return next(model.parameters()).device


def main() -> None:
    args = parse_args()
    prompt_template = Path(args.prompt_template).read_text(encoding="utf-8")
    prompt = prompt_template.replace("{question}", args.question)

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    if args.no_chat_template:
        encoded = tokenizer(prompt, return_tensors="pt")
    else:
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    device = first_model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_len = encoded["input_ids"].shape[-1]

    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_token_ids.append(im_end_id)
    eos_token_ids = sorted(set(token_id for token_id in eos_token_ids if token_id is not None))

    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_ids,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    print(response.strip())


if __name__ == "__main__":
    main()
