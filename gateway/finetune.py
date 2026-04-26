"""
X25 Fine-tuning Pipeline — Phase 5.

What gets fine-tuned and why:

  We fine-tune a ROUTING CLASSIFIER — a lightweight model that learns
  to predict which tier (slm/mid/frontier) is optimal for a given prompt,
  specific to this org's task mix and quality preferences.

  This is different from fine-tuning the LLM itself. We're training
  X25's brain on real observations from this org's call history.

Two fine-tuning paths:

  Path A — Routing classifier (auto, Stage 4):
    Data:   (prompt_text*, task_type, quality_score, tier_used, reward)
    Model:  Llama 3.2 3B → LoRA fine-tune → routing head
    Output: A model that classifies "which tier for this prompt?" with
            org-specific accuracy, registered as a custom SLM in the pool.

  Path B — Domain SLM (Stage 3 feedback):
    Data:   Org-submitted (prompt, good_model) examples
    Model:  Llama 3.2 3B → LoRA fine-tune → domain-adapted responder
    Output: A cheap model tuned to this org's specific task vocabulary.

  * Raw prompts stored only when org opts in at Stage 4.

Training backend:
  Local GPU:   Unsloth (2× faster, 70% less VRAM — runs on RTX 3060+)
  Cloud:       Google Colab (free T4, ~45 min per 200 examples)
  API-based:   Together AI fine-tuning API (no GPU needed, $1–3 per run)

Reference: Unsloth (https://unsloth.ai), LoRA (Hu et al. 2021, arXiv:2106.09685)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Optional

JOBS_DB    = os.environ.get("X25_FINETUNE_DB",  "/tmp/x25_finetune.db")
DATA_DIR   = os.environ.get("X25_FINETUNE_DATA", "/tmp/x25_finetune_data")
MODELS_DIR = os.environ.get("X25_CUSTOM_MODELS", "/tmp/x25_custom_models")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class FinetuneJob:
    job_id:       str
    org:          str
    path:         str        # "routing_classifier" or "domain_slm"
    status:       str        # pending / preparing / training / complete / failed
    base_model:   str
    data_path:    str
    output_path:  str
    n_examples:   int
    created_at:   float
    started_at:   float
    finished_at:  float
    model_id:     str        # registered model ID when complete
    error:        str


class FinetuneStore:
    """SQLite-backed job tracker."""

    def __init__(self, db_path: str = JOBS_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS finetune_jobs (
                    job_id       TEXT PRIMARY KEY,
                    org          TEXT NOT NULL,
                    path         TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    base_model   TEXT NOT NULL,
                    data_path    TEXT NOT NULL DEFAULT '',
                    output_path  TEXT NOT NULL DEFAULT '',
                    n_examples   INTEGER NOT NULL DEFAULT 0,
                    created_at   REAL NOT NULL,
                    started_at   REAL NOT NULL DEFAULT 0,
                    finished_at  REAL NOT NULL DEFAULT 0,
                    model_id     TEXT NOT NULL DEFAULT '',
                    error        TEXT NOT NULL DEFAULT ''
                )
            """)
            # Custom models registered after fine-tuning
            conn.execute("""
                CREATE TABLE IF NOT EXISTS custom_models (
                    model_id     TEXT PRIMARY KEY,
                    org          TEXT NOT NULL,
                    base_model   TEXT NOT NULL,
                    model_path   TEXT NOT NULL,
                    job_id       TEXT NOT NULL,
                    registered_at REAL NOT NULL,
                    active       INTEGER NOT NULL DEFAULT 1
                )
            """)
            conn.commit()

    def create_job(self, org: str, path: str, base_model: str) -> FinetuneJob:
        job_id = f"ft-{uuid.uuid4().hex[:12]}"
        now    = time.time()
        output = os.path.join(MODELS_DIR, org.replace("/", "_"), job_id)
        data   = os.path.join(DATA_DIR,   org.replace("/", "_"), f"{job_id}.jsonl")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        os.makedirs(os.path.dirname(data),   exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO finetune_jobs (job_id, org, path, base_model, data_path, "
                "output_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, org, path, base_model, data, output, now),
            )
            conn.commit()

        return self._get(job_id)

    def update(self, job_id: str, **kwargs):
        fields = ", ".join(f"{k}=?" for k in kwargs)
        values = list(kwargs.values()) + [job_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE finetune_jobs SET {fields} WHERE job_id=?", values)
            conn.commit()

    def _get(self, job_id: str) -> Optional[FinetuneJob]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT job_id,org,path,status,base_model,data_path,output_path,"
                "n_examples,created_at,started_at,finished_at,model_id,error "
                "FROM finetune_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if not row:
            return None
        return FinetuneJob(*row)

    def get_job(self, job_id: str) -> Optional[FinetuneJob]:
        return self._get(job_id)

    def list_jobs(self, org: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT job_id, path, status, n_examples, created_at, finished_at, model_id "
                "FROM finetune_jobs WHERE org=? ORDER BY created_at DESC",
                (org,),
            ).fetchall()
        return [
            {"job_id": r[0], "path": r[1], "status": r[2], "n_examples": r[3],
             "created_at": r[4], "finished_at": r[5], "model_id": r[6]}
            for r in rows
        ]

    def register_model(self, org: str, model_id: str, base_model: str,
                       model_path: str, job_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO custom_models "
                "(model_id, org, base_model, model_path, job_id, registered_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (model_id, org, base_model, model_path, job_id, time.time()),
            )
            conn.commit()

    def list_custom_models(self, org: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT model_id, base_model, model_path, registered_at, active "
                "FROM custom_models WHERE org=? ORDER BY registered_at DESC",
                (org,),
            ).fetchall()
        return [
            {"model_id": r[0], "base_model": r[1], "model_path": r[2],
             "registered_at": r[3], "active": bool(r[4])}
            for r in rows
        ]


class FinetuneManager:
    """
    Orchestrates the full fine-tuning pipeline for an org.

    Step 1: Extract training data from audit log + feedback table
    Step 2: Format as instruction-tuning JSONL
    Step 3: Generate Unsloth training script
    Step 4: Run training (local GPU / Colab / Together AI)
    Step 5: Register the fine-tuned model back into the routing pool
    """

    BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self):
        self.store = FinetuneStore()

    def prepare_routing_data(self, org: str, min_quality: float = 0.7) -> list[dict]:
        """
        Extract routing training examples from audit log.
        Each example: (prompt_hash, task_type, optimize_for, best_tier, reward)

        Note: we use task_type as a proxy for the prompt since raw prompts
        are hashed for privacy. At Stage 4, orgs can opt in to prompt logging.
        """
        from audit import AuditStore
        records = AuditStore().get_recent(org=org, limit=2000)

        examples = []
        for r in records:
            if r["quality_score"] < min_quality:
                continue
            gm = r.get("goal_match", {})
            examples.append({
                "task_type":    r["task_type"],
                "optimize_for": r["optimize_for"],
                "selected_tier": r["selected_tier"],
                "quality_score": r["quality_score"],
                "reward":        gm.get("overall_reward", 0.0),
                "cost_saved":    gm.get("cost", 0.0),
                "cascade_steps": r["cascade_steps"],
            })
        return examples

    def prepare_feedback_data(self, org: str) -> list[dict]:
        """Extract Stage 3 labelled examples submitted by the org."""
        from stages import get_stage_tracker
        with sqlite3.connect(get_stage_tracker().db_path) as conn:
            rows = conn.execute(
                "SELECT prompt, good_model FROM org_feedback WHERE org=?", (org,)
            ).fetchall()
        return [{"prompt": r[0], "good_model": r[1]} for r in rows]

    def format_as_jsonl(self, examples: list[dict], path: str) -> int:
        """
        Convert routing examples to instruction-tuning JSONL.
        Format: {"instruction": ..., "input": ..., "output": ...}
        """
        written = 0
        with open(path, "w") as f:
            for ex in examples:
                cost_w    = ex["optimize_for"].get("cost", 0.33)
                quality_w = ex["optimize_for"].get("quality", 0.34)
                latency_w = ex["optimize_for"].get("latency", 0.33)
                instruction = (
                    f"You are an LLM routing classifier. Given a task type and "
                    f"optimization preferences, predict the optimal model tier."
                )
                input_text = (
                    f"Task type: {ex['task_type']}\n"
                    f"Optimize for: cost={cost_w:.2f}, quality={quality_w:.2f}, "
                    f"latency={latency_w:.2f}\n"
                    f"Quality score achieved: {ex['quality_score']:.2f}\n"
                    f"Cost saved vs frontier: {ex['cost_saved']:.2f}"
                )
                output_text = (
                    f"optimal_tier: {ex['selected_tier']}\n"
                    f"reward: {ex['reward']:.3f}"
                )
                f.write(json.dumps({
                    "instruction": instruction,
                    "input":       input_text,
                    "output":      output_text,
                }) + "\n")
                written += 1
        return written

    def generate_training_script(self, job: FinetuneJob) -> str:
        """
        Generate a complete Unsloth LoRA training script.
        Runnable on: local GPU (RTX 3060+) or Google Colab T4.
        """
        script = f'''"""
X25 Fine-tuning Script — Auto-generated for org: {job.org}
Job ID: {job.job_id}

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes

Run:
    python {job.job_id}_train.py
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ── Model config ─────────────────────────────────────────
MAX_SEQ_LENGTH = 2048
DTYPE          = None          # auto-detect (float16 on T4, bfloat16 on A100)
LOAD_IN_4BIT   = True          # 4-bit quantisation — fits on 8GB VRAM

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "{self.BASE_MODEL}",
    max_seq_length= MAX_SEQ_LENGTH,
    dtype         = DTYPE,
    load_in_4bit  = LOAD_IN_4BIT,
)

# ── LoRA config ───────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                   = 16,       # LoRA rank — higher = more params, better quality
    target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    lora_alpha          = 16,
    lora_dropout        = 0,
    bias                = "none",
    use_gradient_checkpointing = "unsloth",
    random_state        = 42,
)

# ── Data ──────────────────────────────────────────────────
ALPACA_PROMPT = """Below is an instruction that describes a task. Write a response.

### Instruction:
{{}}

### Input:
{{}}

### Response:
{{}}"""

def format_prompt(examples):
    texts = []
    for inst, inp, out in zip(examples["instruction"],
                               examples["input"],
                               examples["output"]):
        texts.append(ALPACA_PROMPT.format(inst, inp, out) + tokenizer.eos_token)
    return {{"text": texts}}

dataset = load_dataset("json", data_files="{job.data_path}", split="train")
dataset = dataset.map(format_prompt, batched=True)

# ── Training ─────────────────────────────────────────────
trainer = SFTTrainer(
    model        = model,
    tokenizer    = tokenizer,
    train_dataset= dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps                = 5,
        num_train_epochs            = 3,
        learning_rate               = 2e-4,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        output_dir                  = "{job.output_path}",
        report_to                   = "none",
    ),
)

trainer.train()

# ── Save ─────────────────────────────────────────────────
model.save_pretrained("{job.output_path}/lora_weights")
tokenizer.save_pretrained("{job.output_path}/lora_weights")
print("Training complete. Model saved to: {job.output_path}/lora_weights")
print("Register with X25:")
print("  POST /improve/{job.org}/register")
print("  {{\\"job_id\\": \\"{job.job_id}\\", \\"model_path\\": \\"{job.output_path}/lora_weights\\"}}")
'''
        return script

    def start_job(self, org: str, path: str = "routing_classifier",
                  dry_run: bool = False) -> FinetuneJob:
        """
        Kick off a fine-tuning job.

        dry_run=True: prepares data + generates script but doesn't train.
        dry_run=False: runs training if GPU available, else prints Colab instructions.
        """
        job = self.store.create_job(org=org, path=path, base_model=self.BASE_MODEL)
        self.store.update(job.job_id, status="preparing", started_at=time.time())

        # Prepare data
        if path == "routing_classifier":
            examples = self.prepare_routing_data(org)
        else:
            examples = self.prepare_feedback_data(org)

        if not examples:
            self.store.update(job.job_id, status="failed",
                              error="No training examples found.")
            return self.store.get_job(job.job_id)

        n = self.format_as_jsonl(examples, job.data_path)
        self.store.update(job.job_id, status="data_ready", n_examples=n)

        # Generate training script
        script_path = job.data_path.replace(".jsonl", "_train.py")
        script      = self.generate_training_script(self.store.get_job(job.job_id))
        with open(script_path, "w") as f:
            f.write(script)

        if dry_run:
            self.store.update(job.job_id, status="script_ready")
            return self.store.get_job(job.job_id)

        # Try local GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.store.update(job.job_id, status="training")
                import subprocess
                subprocess.run(["python", script_path], check=True)
                self.store.update(job.job_id, status="complete",
                                  finished_at=time.time())
            else:
                self.store.update(job.job_id, status="awaiting_gpu")
        except Exception as e:
            self.store.update(job.job_id, status="awaiting_gpu",
                              error=str(e)[:200])

        return self.store.get_job(job.job_id)

    def register_complete_model(self, org: str, job_id: str,
                                model_path: str) -> str:
        """
        Called after training completes (locally or on Colab).
        Registers the fine-tuned model into the routing pool.
        """
        from model_registry import get_registry, ModelEntry
        import time as t

        model_id = f"x25-custom/{org.replace('/', '-')}-{job_id[:8]}"

        # Register in finetune store
        self.store.register_model(
            org=org, model_id=model_id,
            base_model=self.BASE_MODEL,
            model_path=model_path,
            job_id=job_id,
        )
        self.store.update(job_id, status="complete", model_id=model_id,
                          finished_at=time.time())

        # Register as a custom SLM in the model registry
        entry = ModelEntry(
            id=model_id,
            name=f"X25 Custom ({org})",
            provider="x25-custom",
            tier="slm",
            cost_per_1m_input=0.01,    # near-zero cost — local inference
            cost_per_1m_output=0.01,
            context_length=8192,
            is_vision=False,
            energy_j_per_token=0.3,    # smaller than any cloud model
            created=int(t.time()),
        )
        registry = get_registry()
        registry._all_models.append(entry)
        print(f"[finetune] registered custom model '{model_id}' for org '{org}'")
        return model_id


# ── Singleton ──────────────────────────────────────────────────────────────────

_manager: Optional[FinetuneManager] = None


def get_finetune_manager() -> FinetuneManager:
    global _manager
    if _manager is None:
        _manager = FinetuneManager()
    return _manager
