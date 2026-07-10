"""Pipeline 3 (learning add-on): full fine-tune of DeBERTa-v3-base.

WHAT THIS IS
------------
Pipelines 1 and 2 keep their feature extractor *frozen*: BGE turns a comment
into a fixed 384-dim embedding, and only a small classifier (LogReg / LightGBM)
is trained on top. That is called a **linear probe** -- the language model never
learns anything about *toxicity*, it just describes the sentence and a cheap
model draws a line through those descriptions.

This script does the "proper" end-to-end upgrade: it takes a full transformer
encoder (DeBERTa-v3-base), bolts a 2-class classification head on top, and
**trains the whole thing** -- encoder weights included -- on the toxicity labels.
Now the representation itself adapts to the task instead of being fixed.

Trade-off: far more capacity (and more risk of overfitting on a small dataset),
much heavier to train, and a ~440 MB model to serve instead of a 4 KB joblib.
On this tiny 100-row toy dataset it may or may not beat the 0.95 BGE+LogReg
baseline -- the real point is to learn the workflow so it transfers to a bigger,
messier problem (thousands of rows, longer text), which is why we use `base`
(not `small`) and a generous `MAX_LENGTH`.

THE PIPELINE, STEP BY STEP
--------------------------
    1. load_split()          CSV  -> (texts, integer labels)
    2. build_dataset()       texts -> tokenised 🤗 Dataset (input_ids, ...)
    3. build_model()         load DeBERTa-v3-base + a fresh 2-class head
    4. compute_metrics()     logits -> accuracy + macro-F1
    5. train()               Trainer runs the fine-tune loop, picks best epoch
    6. main()                wires it together, evaluates, saves the model

PREREQUISITES
-------------
Beyond the project's usual deps, a full fine-tune needs three more packages
(DeBERTa-v3's tokenizer is SentencePiece-based, hence `sentencepiece`):

    pip install datasets accelerate sentencepiece

The first run downloads `microsoft/deberta-v3-base` (~440 MB) from the Hugging
Face Hub, so it needs network access once (unlike the offline BGE pipeline).

RUN
---
    python finetune_deberta.py

Heads up: a full fine-tune of a 184M-param model is *slow on CPU* (minutes per
run). A GPU is dramatically faster -- Trainer uses one automatically if torch
sees CUDA. The fine-tuned model is written to `models_out/deberta-v3-toxic/`,
which .gitignore excludes (too big to commit -- unlike the tiny logreg.joblib).

NOTE ON EVALUATION
------------------
For simplicity this mirrors classify_logreg.py: it trains on the training split
and reports on eval_dataset.csv, also using that eval set to pick the best epoch
(early stopping). That double-use is a mild optimism leak. For a rigorous number
on a small dataset, use k-fold instead (see classify_logreg_kfold.py for the
pattern); here we favour a readable single-split run.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from dataset import EVAL_CSV, LABELS, ROOT, TRAINING_CSV, load_dataset

# --- Configuration knobs (the decisions we made, in one place) ---------------
MODEL_NAME = "microsoft/deberta-v3-base"   # `base` so it transfers to the real task
# MODEL_NAME = str(ROOT / "models" / "deberta-v3-base")
MODEL_OUT = ROOT / "models_out" / "deberta-v3-toxic"   # saved model (gitignored)
CHECKPOINT_DIR = ROOT / "deberta_checkpoints"          # Trainer scratch (gitignored)

MAX_LENGTH = 256          # comments are short; roomy for a real task's longer text
LEARNING_RATE = 2e-5      # encoders want a *small* LR or they forget pretraining
EPOCHS = 5                # early stopping usually halts before this on tiny data
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01       # mild regularisation -- helps on small data
SEED = 42

# Map the string labels to the integer class ids the model outputs, and back.
# e.g. {"toxic": 0, "non_toxic": 1} and {0: "toxic", 1: "non_toxic"}.
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# --- Step 1: data -> (texts, integer labels) ---------------------------------
def load_split(csv_path):
    """Load a `label,text` CSV and turn the string labels into integer ids.

    The transformer's classification head predicts an integer class (0/1), so we
    encode "toxic"/"non_toxic" using the shared LABEL2ID map.
    """
    data = load_dataset(csv_path)
    labels = [LABEL2ID[label] for label in data.labels]
    print(f"Loaded {len(data)} rows from {csv_path.name}")
    return data.texts, labels


# --- Step 2: texts -> tokenised 🤗 Dataset -----------------------------------
def build_dataset(texts, labels, tokenizer):
    """Tokenise raw text into the input_ids / attention_mask the model consumes.

    A transformer can't read strings -- the tokenizer splits each comment into
    sub-word pieces and maps them to integer ids. We return a 🤗 `Dataset`, which
    is what `Trainer` expects. Padding is done later, per-batch, by the collator
    (more efficient than padding every row to MAX_LENGTH up front).
    """
    from datasets import Dataset as HFDataset

    ds = HFDataset.from_dict({"text": texts, "label": labels})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    # batched=True tokenises many rows per call (faster); drop the raw text after.
    return ds.map(tokenize, batched=True, remove_columns=["text"])


# --- Step 3: load the encoder + a fresh classification head -------------------
def build_model():
    """Load DeBERTa-v3-base with a new, randomly-initialised 2-class head.

    `from_pretrained` brings in the encoder's pretrained weights (its language
    understanding). The classification head on top starts random -- training
    teaches it to map the encoder's output to toxic / non_toxic, and (because
    this is a *full* fine-tune) also nudges the encoder weights themselves.
    """
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


# --- Step 4: logits -> the metrics we care about -----------------------------
def compute_metrics(eval_pred):
    """Turn raw model outputs into accuracy + macro-F1 for each eval pass.

    Trainer hands us the logits (one score per class) and the true labels. We
    take the arg-max as the prediction. macro-F1 weights both classes equally,
    which is the more honest summary on a small, roughly balanced set -- the same
    metric the LogReg k-fold script reports, so the numbers are comparable.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# --- Step 5: the fine-tuning loop --------------------------------------------
def train(model, tokenizer, train_ds, eval_ds):
    """Run the fine-tune with 🤗 Trainer and return the trained Trainer.

    TrainingArguments are the recipe (LR, epochs, batch size, regularisation).
    `load_best_model_at_end` + EarlyStoppingCallback mean we keep the epoch with
    the best eval macro-F1 rather than the last one -- important on small data
    where later epochs tend to overfit.
    """
    from transformers import (
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    # Dynamic padding: pad each batch to its own longest row, not to MAX_LENGTH.
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.1,               # ease the LR up to avoid an early shock
        eval_strategy="epoch",          # evaluate after every epoch...
        save_strategy="epoch",          # ...and checkpoint, so we can pick the best
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,             # don't fill the disk with checkpoints
        logging_strategy="epoch",
        report_to="none",               # no wandb/tensorboard for a toy run
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,     # transformers v5 name (was `tokenizer=`)
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    return trainer


# --- Step 6: orchestrate everything ------------------------------------------
def main():
    from transformers import AutoTokenizer, set_seed

    set_seed(SEED)  # make the run reproducible (weights init, shuffling, etc.)

    # 1. Data.
    train_texts, train_labels = load_split(TRAINING_CSV)
    eval_texts, eval_labels = load_split(EVAL_CSV)

    # 2. Tokeniser + tokenised datasets.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = build_dataset(train_texts, train_labels, tokenizer)
    eval_ds = build_dataset(eval_texts, eval_labels, tokenizer)

    # 3 + 5. Model, then fine-tune.
    print(f"\nLoading {MODEL_NAME} and fine-tuning (this is slow on CPU)...\n")
    model = build_model()
    trainer = train(model, tokenizer, train_ds, eval_ds)

    # Report the best epoch's scores on the eval set.
    metrics = trainer.evaluate()
    print("\nEval results (best epoch):")
    print(f"  accuracy: {metrics['eval_accuracy']:.3f}")
    print(f"  f1_macro: {metrics['eval_f1_macro']:.3f}")
    print("\n(Compare against BGE+LogReg's ~0.95 -- and remember eval is only 20 rows.)")

    # 6. Save the fine-tuned model + tokenizer so it can be loaded for serving.
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))
    print(f"\nSaved fine-tuned model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
