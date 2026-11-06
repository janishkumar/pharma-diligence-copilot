#!/usr/bin/env python3
"""M7: QLoRA fine-tune of Qwen2.5-7B-Instruct with Unsloth (PRD 11.3).

Runs on the GPU VM. Inputs: train_sft.jsonl / val_sft.jsonl (messages format).
Completion-only loss (assistant turn only) so capacity isn't spent on
reproducing the retrieved context. Resumable from the latest checkpoint.

PRD deviations (documented in CHANGELOG): max_seq_length 4096 (3 numbered
800-token chunks do not fit in 2048), packing off (incompatible with
completion-only loss, and sequences are mostly full anyway).
"""
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer

BASE = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DATA = Path(os.getenv("DATA_DIR", "data"))
OUT = Path(os.getenv("OUT_DIR", "out"))
MAX_SEQ = 4096
SEED = 42
MAX_STEPS = int(os.getenv("MAX_STEPS", "-1"))  # >0 for smoke tests


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def make_args():
    import inspect
    kw = dict(
        output_dir=str(OUT / "checkpoints"),
        dataset_text_field="text", packing=False,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,  # effective 16
        num_train_epochs=2, max_steps=MAX_STEPS, learning_rate=2e-4, lr_scheduler_type="cosine",
        warmup_ratio=0.05, weight_decay=0.01, optim="adamw_8bit",
        bf16=True, fp16=False, seed=SEED, logging_steps=5,
        eval_strategy="steps", eval_steps=25 if MAX_STEPS < 0 else 3,
        save_strategy="steps", save_steps=50 if MAX_STEPS < 0 else 3,
        save_total_limit=2, report_to="tensorboard", logging_dir=str(OUT / "tb"),
        per_device_eval_batch_size=2, dataset_num_proc=2,
    )
    params = inspect.signature(SFTConfig.__init__).parameters
    kw["max_length" if "max_length" in params and "max_seq_length" not in params else "max_seq_length"] = MAX_SEQ
    return SFTConfig(**kw)


def main():
    assert torch.cuda.is_available(), "no CUDA device"
    print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )

    def to_text(recs):
        return Dataset.from_list([
            {"text": tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)}
            for r in recs
        ])

    train_ds, val_ds = to_text(load(DATA / "train_sft.jsonl")), to_text(load(DATA / "val_sft.jsonl"))
    lens = [len(tok(t).input_ids) for t in train_ds["text"][:200]]
    print(f"train {len(train_ds)} | val {len(val_ds)} | sample token len mean {sum(lens)/len(lens):.0f} max {max(lens)}")
    over = sum(1 for t in train_ds["text"] if len(tok(t).input_ids) > MAX_SEQ)
    print("examples over max_seq (would be truncated):", over)

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=train_ds, eval_dataset=val_ds,
        args=make_args(),
    )
    # Completion-only loss: mask everything except the assistant turn (ChatML).
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n",
    )

    ckpts = sorted((OUT / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) \
        if (OUT / "checkpoints").exists() else []
    if ckpts:
        print("resuming from", ckpts[-1])
    result = trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    print("TRAIN_RESULT", json.dumps({k: v for k, v in result.metrics.items()}))

    final = trainer.evaluate()
    print("FINAL_EVAL", json.dumps(final))

    model.save_pretrained(str(OUT / "adapter"))
    tok.save_pretrained(str(OUT / "adapter"))
    (OUT / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print("ADAPTER_SAVED", OUT / "adapter")


if __name__ == "__main__":
    sys.exit(main())
