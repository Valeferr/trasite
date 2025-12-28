import os
import warnings
import inspect
from enum import Enum

import numpy as np
import pandas as pd
import torch

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import Dataset

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ResampleStrategy(Enum):
    NONE = "none"
    OVER = "over"
    UNDER = "under"


def build_training_args(**kwargs) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__).parameters

    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in sig and "eval_strategy" in sig:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    if "eval_strategy" in kwargs and "eval_strategy" not in sig and "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    filtered = {k: v for k, v in kwargs.items() if k in sig}
    return TrainingArguments(**filtered)


def dataset_from_pandas_safe(df: pd.DataFrame) -> Dataset:
    try:
        return Dataset.from_pandas(df, preserve_index=False)
    except TypeError:
        return Dataset.from_pandas(df)


def predict_compat(trainer: Trainer, dataset: Dataset):
    sig = inspect.signature(trainer.predict).parameters
    if "test_dataset" in sig:
        return trainer.predict(test_dataset=dataset)
    return trainer.predict(dataset)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def report_split(trainer: Trainer, dataset: Dataset, split_name: str, label_names=None):
    if label_names is None:
        label_names = ["Fraud(0)", "Legit(1)"]

    out = predict_compat(trainer, dataset)

    logits = out.predictions
    y_true = out.label_ids
    y_pred = np.argmax(logits, axis=-1)

    hf_metrics = dict(out.metrics) if out.metrics is not None else {}
    try:
        probs = softmax_np(logits, axis=-1)
        auc = float(roc_auc_score(y_true, probs[:, 1]))
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    print(f"\n================ {split_name.upper()} RESULTS ================")
    print("Trainer metrics:")
    for k in sorted(hf_metrics.keys()):
        print(f"  {k}: {hf_metrics[k]}")
    print(f"Extra ROC-AUC (from probs): {auc}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nConfusion matrix normalized by true class (rows sum to 1):")
    print(np.round(cm_norm, 4))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "cm_00": int(cm[0, 0]),
        "cm_01": int(cm[0, 1]),
        "cm_10": int(cm[1, 0]),
        "cm_11": int(cm[1, 1]),
    }


class FraudDetectorTrainer:
    def __init__(self, model_name, base_output_dir="./backend/data/trained"):
        self.model_name = model_name
        safe_model_name = model_name.replace("/", "_")
        self.output_dir = os.path.join(base_output_dir, safe_model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        acc = accuracy_score(labels, preds)

        try:
            probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
            auc = roc_auc_score(labels, probs[:, 1])
        except Exception:
            auc = 0.0

        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "roc_auc": auc}

    @staticmethod
    def _apply_resampling(
        train_df: pd.DataFrame,
        strategy: ResampleStrategy,
        seed: int = 42,
        fraud_to_legit_ratio: float = 0.5,
    ) -> pd.DataFrame:
        if strategy == ResampleStrategy.NONE:
            return train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        if fraud_to_legit_ratio is None:
            fraud_to_legit_ratio = 0.5

        r = float(fraud_to_legit_ratio)
        if not np.isfinite(r) or r <= 0:
            raise ValueError("fraud_to_legit_ratio must be a finite number > 0 (e.g., 0.5 for 1:2, 1.0 for 1:1).")

        X = train_df[["review"]]
        y = train_df["labels"]

        n_fraud = int((y == 0).sum())
        n_legit = int((y == 1).sum())

        if n_fraud == 0 or n_legit == 0:
            return train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        if strategy == ResampleStrategy.OVER:
            if (n_fraud / n_legit) < r:
                target_fraud = int(np.ceil(r * n_legit))
                target_legit = n_legit
            else:
                target_fraud = n_fraud
                target_legit = int(np.ceil(n_fraud / r))

            sampling_strat = {0: max(n_fraud, target_fraud), 1: max(n_legit, target_legit)}
            sampler = RandomOverSampler(sampling_strategy=sampling_strat, random_state=seed)

        elif strategy == ResampleStrategy.UNDER:
            if (n_fraud / n_legit) < r:
                target_fraud = n_fraud
                target_legit = int(np.floor(n_fraud / r))
            else:
                target_legit = n_legit
                target_fraud = int(np.floor(r * n_legit))

            sampling_strat = {0: min(n_fraud, target_fraud), 1: min(n_legit, target_legit)}
            sampler = RandomUnderSampler(sampling_strategy=sampling_strat, random_state=seed)

        else:
            return train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        X_res, y_res = sampler.fit_resample(X, y)
        out = pd.concat([X_res, y_res], axis=1)
        return out.sample(frac=1, random_state=seed).reset_index(drop=True)

    def train_cross_validation(
        self,
        csv_path: str,
        resample_strategy: ResampleStrategy = ResampleStrategy.NONE,
        seed: int = 42,
        fraud_to_legit_ratio: float = 0.5,
    ):
        set_seed(seed)
        np.random.seed(seed)

        if isinstance(resample_strategy, str):
            resample_strategy = ResampleStrategy(resample_strategy.lower().strip())

        print(f" TRAINING MODEL: {self.model_name}")
        print(f"Loading data from {csv_path}...")

        df = pd.read_csv(csv_path).dropna(subset=["review", "legit"])
        initial_len = len(df)
        df = df.drop_duplicates(subset=["review"])
        print(f"Dropped {initial_len - len(df)} duplicate reviews.")
        print(f"Raw 'legit' values found: {df['legit'].unique()}")

        def map_labels(x):
            if x == -1:
                return 0
            if x == 0:
                return 0
            return 1

        df["labels"] = df["legit"].apply(map_labels)
        print(f"Mapped distribution (0=Fraud, 1=Legit): {df['labels'].value_counts().to_dict()}")

        if df["labels"].nunique() < 2:
            print("CRITICAL ERROR: Only one class found. Cannot train classifier.")
            return

        df = df[["review", "labels"]].reset_index(drop=True)

        train_val_df, test_df = train_test_split(
            df,
            test_size=0.20,
            random_state=seed,
            shuffle=True,
            stratify=df["labels"],
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.125,
            random_state=seed,
            shuffle=True,
            stratify=train_val_df["labels"],
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        print(f"\nSplit sizes -> train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
        print(f"Train distribution (pre-resample): {train_df['labels'].value_counts().to_dict()}")
        print(f"Val   distribution: {val_df['labels'].value_counts().to_dict()}")
        print(f"Test  distribution: {test_df['labels'].value_counts().to_dict()}")
        print(f"Resampling strategy: {resample_strategy.value}")
        print(f"Fraud:Legit target ratio: {fraud_to_legit_ratio}:1 (Fraud/Legit={fraud_to_legit_ratio})")

        train_df = self._apply_resampling(
            train_df,
            resample_strategy,
            seed=seed,
            fraud_to_legit_ratio=fraud_to_legit_ratio,
        )
        print(f"Train distribution (post-resample): {train_df['labels'].value_counts().to_dict()}")

        def tokenize_function(examples):
            return self.tokenizer(examples["review"], truncation=True, padding=False, max_length=512)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        train_dataset = dataset_from_pandas_safe(train_df)
        val_dataset = dataset_from_pandas_safe(val_df)
        test_dataset = dataset_from_pandas_safe(test_df)

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)
        tokenized_test = test_dataset.map(tokenize_function, batched=True)

        for name in ["tokenized_train", "tokenized_val", "tokenized_test"]:
            ds = locals()[name]
            try:
                locals()[name] = ds.remove_columns(["review"])
            except Exception:
                pass

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(self.device)

        if "xsmall" in self.model_name:
            batch_size = 16
            grad_accum = 2
        else:
            batch_size = 8
            grad_accum = 4

        use_cuda = torch.cuda.is_available()
        optim_choice = "adamw_torch_fused" if use_cuda else "adamw_torch"

        os.makedirs(self.output_dir, exist_ok=True)

        training_args = build_training_args(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=3,
            weight_decay=0.01,
            fp16=use_cuda,
            optim=optim_choice,
            group_by_length=True,
            dataloader_num_workers=0,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=2,
            push_to_hub=False,
            report_to="none",
        )

        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        if "tokenizer" in inspect.signature(Trainer.__init__).parameters:
            trainer_kwargs["tokenizer"] = self.tokenizer

        trainer = Trainer(**trainer_kwargs)

        trainer.train()

        label_names = ["Fraud(0)", "Legit(1)"]
        val_summary = report_split(trainer, tokenized_val, "validation", label_names=label_names)
        test_summary = report_split(trainer, tokenized_test, "test", label_names=label_names)

        print(f"\nSaving final model for {self.model_name} to {self.output_dir} ...")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        row = {
            "model": self.model_name,
            "resample_strategy": resample_strategy.value,
            "fraud_to_legit_ratio": float(fraud_to_legit_ratio),
            **{f"val_{k}": v for k, v in val_summary.items()},
            **{f"test_{k}": v for k, v in test_summary.items()},
        }
        pd.DataFrame([row]).to_csv(os.path.join(self.output_dir, "results_70_10_20.csv"), index=False)

        del model
        del trainer
        if use_cuda:
            torch.cuda.empty_cache()

    def predict_from_csv(self, input_csv: str, output_csv: str):
        if not os.path.exists(self.output_dir):
            raise ValueError(f"Model not found at {self.output_dir}")

        print(f"Running inference using {self.model_name}...")

        clf_pipe = pipeline(
            "text-classification",
            model=self.output_dir,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512,
        )

        df = pd.read_csv(input_csv)
        texts = df["review"].astype(str).tolist()

        predictions = clf_pipe(texts, batch_size=32)

        final_labels, final_scores = [], []
        for p in predictions:
            label_str = p["label"]
            if "LABEL_" in label_str:
                label_id = int(label_str.split("_")[-1])
            else:
                label_id = 1 if label_str.lower() in ["legit", "1", "pos"] else 0

            final_labels.append(label_id)
            final_scores.append(p["score"])

        df["predicted_label"] = final_labels
        df["prediction_confidence"] = final_scores
        df.to_csv(output_csv, index=False)

        print(f"Saved predictions to {output_csv}")


def main():
    train_file = "./backend/data/yelp_dataset.csv"
    predict_file = "./backend/data/reviews_en_clean.csv"

    models_to_run = [
        "distilroberta-base",
        "microsoft/deberta-v3-xsmall",
    ]

    resampling = ResampleStrategy.UNDER

    for model_name in models_to_run:
        if os.path.exists(train_file):
            detector = FraudDetectorTrainer(model_name=model_name)

            detector.train_cross_validation(
                train_file,
                resample_strategy=resampling,
                seed=42,
                fraud_to_legit_ratio=1.0,
            )

            if os.path.exists(predict_file):
                safe_name = model_name.replace("/", "_")
                output_file = f"./backend/data/reviews_classified_{safe_name}_3.csv"
                detector.predict_from_csv(predict_file, output_file)
        else:
            print(f"Training file {train_file} not found.")


if __name__ == "__main__":
    main()
