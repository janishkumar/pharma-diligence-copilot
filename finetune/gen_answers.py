#!/usr/bin/env python3
"""Generate answers on the test prompts with the BASE model and the
FINE-TUNED model (adapter on/off), greedy decoding, same prompts.

Runs on the GPU VM. Writes out/answers_{base,tuned}.jsonl. Judging happens
locally (finetune/eval_models.py) so no API keys ever sit on the VM.
"""
import json
import os
from pathlib import Path

import torch
from unsloth import FastLanguageModel

BASE = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DATA = Path(os.getenv("DATA_DIR", "data"))
OUT = Path(os.getenv("OUT_DIR", "out"))
MAX_SEQ = 4096
BATCH = 4
MAX_NEW = 512


def run(model, tok, recs, tag):
    path = OUT / f"answers_{tag}.jsonl"
    done = {json.loads(l)["i"] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [(i, r) for i, r in enumerate(recs) if i not in done]
    tok.padding_side = "left"
    with open(path, "a") as f:
        for b in range(0, len(todo), BATCH):
            batch = todo[b:b + BATCH]
            prompts = [tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True)
                       for _, r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for (i, r), g in zip(batch, gen):
                text = tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                f.write(json.dumps({"i": i, "tag": tag, "answer": text, "meta": r["meta"]}) + "\n")
            f.flush()
            print(f"[{tag}] {min(b + BATCH, len(todo))}/{len(todo)}", flush=True)


def main():
    recs = [json.loads(l) for l in (DATA / "test_sft.jsonl").read_text().splitlines() if l.strip()]
    model, tok = FastLanguageModel.from_pretrained(model_name=BASE, max_seq_length=MAX_SEQ, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    run(model, tok, recs, "base")                      # base model, no adapter

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(OUT / "adapter"))
    FastLanguageModel.for_inference(model)
    run(model, tok, recs, "tuned")                     # base + fine-tuned adapter
    print("GEN_DONE")


if __name__ == "__main__":
    main()
