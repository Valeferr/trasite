import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import warnings

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    pipeline,
    EarlyStoppingCallback
)
from datasets import Dataset

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        
        try:
            probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
            auc = roc_auc_score(labels, probs[:, 1])
        except Exception:
            auc = 0.0
            
        return {
            'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'roc_auc': auc
        }

    def train_cross_validation(self, csv_path: str, k_folds: int = 3):
        print(f" TRAINING MODEL: {self.model_name}")
        
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        df = df.dropna(subset=['review', 'legit'])
        
        print(f"Raw 'legit' values found: {df['legit'].unique()}")


        def map_labels(x):
            if x == -1: return 0
            if x == 0: return 0 
            return 1
            
        df['labels'] = df['legit'].apply(map_labels)
        
        print(f"Mapped distribution (0=Fraud, 1=Legit): {df['labels'].value_counts().to_dict()}")
        
        if len(df['labels'].unique()) < 2:
            print("CRITICAL ERROR: Only one class found. Cannot train classifier.")
            return

        class_counts = df["labels"].value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        n_minority = class_counts[minority_class]
        n_majority = class_counts[majority_class]
        
        target_majority = max(int(n_majority * 0.2), n_minority)
        target_majority = min(target_majority, n_majority)
        
        minority_df = df[df["labels"] == minority_class]
        majority_df = df[df["labels"] == majority_class]
        
        if target_majority < n_majority:
            print(f"Undersampling majority class to {target_majority}...")
            majority_df = majority_df.sample(n=target_majority, random_state=42)
        
        df = pd.concat([minority_df, majority_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Final Balanced Distribution: {df['labels'].value_counts().to_dict()}")

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'roc_auc': []}

        def tokenize_function(examples):
            return self.tokenizer(examples["review"], truncation=True, padding=False, max_length=512)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['labels'])):
            print(f"\n--- Model: {self.model_name} | Fold {fold + 1}/{k_folds} ---")
            
            fold_output_dir = os.path.join(self.output_dir, f"fold_{fold+1}")
            
            train_df = df.iloc[train_idx]
            eval_df = df.iloc[val_idx]
            
            train_dataset = Dataset.from_pandas(train_df[['review', 'labels']])
            eval_dataset = Dataset.from_pandas(eval_df[['review', 'labels']])

            tokenized_train = train_dataset.map(tokenize_function, batched=True)
            tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            ).to(self.device)

            if "xsmall" in self.model_name:
                batch_size = 16
                grad_accum = 2
            else:
    
                batch_size = 8
                grad_accum = 4
            
            training_args = TrainingArguments(
                output_dir=fold_output_dir,
                
                eval_strategy="epoch",
                save_strategy="epoch",
                
                learning_rate=2e-5,
                
                per_device_train_batch_size=batch_size, 
                per_device_eval_batch_size=batch_size * 2,
                gradient_accumulation_steps=grad_accum,
                
                num_train_epochs=3,
                weight_decay=0.01,
                
                fp16=True,                   
                optim="adamw_torch_fused",   
                group_by_length=True,        
                dataloader_num_workers=4,
                
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                push_to_hub=False,
                report_to="none"
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

            trainer.train()
            
            eval_results = trainer.evaluate()
            print(f"Fold {fold + 1} Results: {eval_results}")
            
            for key in cv_metrics.keys():
                cv_metrics[key].append(eval_results.get(f"eval_{key}", 0.0))
            
            if fold == k_folds - 1:
                print(f"Saving final model for {self.model_name}...")
                trainer.save_model(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)
            
            del model
            del trainer
            torch.cuda.empty_cache()

        avg_metrics = {k: np.mean(v) for k, v in cv_metrics.items()}
        print(f"\n=== Average Results for {self.model_name} ===")
        print(avg_metrics)
        
        results_df = pd.DataFrame([avg_metrics])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        results_df.to_csv(os.path.join(self.output_dir, "cv_results.csv"), index=False)

    def predict_from_csv(self, input_csv: str, output_csv: str):
        if not os.path.exists(self.output_dir):
            raise ValueError(f"Model not found at {self.output_dir}")

        print(f"Running inference using {self.model_name}...")
        
        clf_pipe = pipeline("text-classification", model=self.output_dir, device=0 if torch.cuda.is_available() else -1, truncation=True, max_length=512)
        
        df = pd.read_csv(input_csv)
        texts = df['review'].astype(str).tolist()
        
        predictions = clf_pipe(texts, batch_size=32)
        
        final_labels = []
        final_scores = []
        
        for p in predictions:
            label_str = p['label']
            if "LABEL_" in label_str:
                label_id = int(label_str.split('_')[-1])
            else:
                label_id = 1 if label_str.lower() in ['legit', '1', 'pos'] else 0
                
            final_labels.append(label_id)
            final_scores.append(p['score'])

        df['predicted_label'] = final_labels
        df['prediction_confidence'] = final_scores
        
        df.to_csv(output_csv, index=False)
        print(f"Saved predictions to {output_csv}")

def main(): 
    train_file = "./backend/data/yelp_dataset.csv" 
    predict_file = "./backend/data/reviews_en_clean.csv"

    models_to_run = [
        "distilroberta-base",
        "microsoft/deberta-v3-xsmall"
    ]

    for model_name in models_to_run:
        if os.path.exists(train_file):
            detector = FraudDetectorTrainer(model_name=model_name)
            detector.train_cross_validation(train_file, k_folds=3)
            
            if os.path.exists(predict_file):
                safe_name = model_name.replace("/", "_")
                output_file = f"./backend/data/reviews_classified_{safe_name}.csv"
                detector.predict_from_csv(predict_file, output_file)
        else:
            print(f"Training file {train_file} not found.")

if __name__ == "__main__":
    main()