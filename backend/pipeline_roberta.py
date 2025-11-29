import os
import joblib
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline
from scipy.special import softmax
from huggingface_hub import InferenceClient

try:
    from transformers import logging
    mpl.use("module://mpl_ascii")
    logging.set_verbosity_error()
except ImportError:
    # Optional: If transformers.logging or mpl_ascii backend is not available,
    # continue without enhanced logging or ASCII plotting.
    pass


class RobertaOpenAIDetector:
    def __init__(self, api_key: str = None):
        self.pipe = pipeline(
            "text-classification",
            model="openai-community/roberta-base-openai-detector",
        )

    def classify(self, text: str) -> dict:
        result = self.pipe(
            text,
            truncation=True,
            max_length=512,
        )[0]
        label = str(result["label"]).lower()
        score = float(result["score"])

        if "real" in label:
            real = score
            fake = 1.0 - score
        else:
            fake = score
            real = 1.0 - score

        return {
            "fake": fake,
            "real": real,
        }

    def classify_batch(self, texts, batch_size: int = 32):
        texts = list(texts)
        total = len(texts)
        results = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            print(f"Processing fake/real {end}/{total}")
            batch = texts[start:end]
            outputs = self.pipe(
                batch,
                truncation=True,
                max_length=512,
            )
            for result in outputs:
                label = str(result["label"]).lower()
                score = float(result["score"])
                if "real" in label:
                    real = score
                    fake = 1.0 - score
                else:
                    fake = score
                    real = 1.0 - score
                results.append(
                    {
                        "fake": fake,
                        "real": real,
                    }
                )
        return results


class RobertaSentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

    def analyze(self, text: str) -> dict:
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            output = self.model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        return {
            "negative": float(scores[0]),
            "neutral": float(scores[1]),
            "positive": float(scores[2]),
        }


class OtisAntiSpamAI:
    def __init__(self, api_key: str):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )
        self.pipe = pipeline(
            "text-classification",
            model="Titeiiko/OTIS-Official-Spam-Model",
        )

    def classify(self, text: str) -> dict:
        x = self.pipe(
            text,
            truncation=True,
            max_length=512,
        )[0]
        return {
            "spam": 1 - x["score"],
            "no_spam": x["score"],
        }

    def classify_batch(self, texts, batch_size: int = 32):
        texts = list(texts)
        total = len(texts)
        results = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            print(f"Processing spam {end}/{total}")
            batch = texts[start:end]
            outputs = self.pipe(
                batch,
                truncation=True,
                max_length=512,
            )
            for out in outputs:
                score = out["score"]
                results.append(
                    {
                        "spam": 1 - score,
                        "no_spam": score,
                    }
                )
        return results


class DownstreamModel:
    """
    Trains, saves, loads, and uses a RandomForest classifier based on transformer model outputs.

    This class is designed to work with features extracted from transformer models (such as fake/real, sentiment, and spam scores)
    and provides methods to train a RandomForest classifier, save and load the trained model, and make predictions.
    """
    def plot(self, data: pd.DataFrame, plot_path: str = None) -> None:
        vc = data["legit"].value_counts()
        vc.plot(kind="bar")
        if plot_path is not None:
            plt.savefig(plot_path)
        plt.show()
        
        plt.close()

    def train(self, data: pd.DataFrame, save_path: str, show_plot: bool = False, plot_path: str = None) -> RandomizedSearchCV:
        data = data.copy()
        class_counts = data["legit"].value_counts()
        if len(class_counts) == 2:
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()
            n_minority = class_counts[minority_class]
            n_majority = class_counts[majority_class]
            target_majority = max(int(n_majority * 0.8), n_minority)
            target_majority = min(target_majority, n_majority)
            minority_df = data[data["legit"] == minority_class]
            majority_df = data[data["legit"] == majority_class]
            if target_majority < n_majority:
                majority_df = majority_df.sample(n=target_majority, random_state=42)
            data = pd.concat([minority_df, majority_df]).sample(frac=1, random_state=42).reset_index(drop=True)

        np.random.seed(42)

        if show_plot:
            self.plot(data, plot_path=plot_path)

        features = [
            "fake",
            "real",
            "negative",
            "neutral",
            "positive",
            "spam",
            "no_spam",
        ]
        X = data[features]
        y = data["legit"]

        rf_grid = RandomForestClassifier(random_state=42)
        gr_space = {
            "max_depth": [3, 5, 7, 10],
            "n_estimators": [100, 200, 300, 400, 500],
            "max_features": ["sqrt", "log2", None],
            "min_samples_leaf": [1, 2, 4],
        }

        scoring = {
            "accuracy": "accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
        }

        grid = RandomizedSearchCV(
            estimator=rf_grid,
            param_distributions=gr_space,
            random_state=42,
            n_iter=30,
            cv=3,
            scoring=scoring,
            refit="accuracy",
            verbose=3,
        )

        grid.fit(X, y)
        print("Random Forest model trained.")
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best cross-validation accuracy: {grid.best_score_}")
        print(f'CV results: {grid.cv_results_}')
        print(f"Feature importances: {grid.best_estimator_.feature_importances_}")
        print("Saving the trained model...")
        self.save(grid, save_path)

        return grid

    def save(self, model, path: str) -> None:
        best_rf_model = model.best_estimator_
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(best_rf_model, path)
        print(f"Model saved as '{path}'")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"The specified path '{path}' does not exist."
            )
        return joblib.load(path)

    def predict_from_csv(
        self,
        model_file_path: str,
        roberta_output_file_path: str,
        train_file_path: str,
    ) -> pd.DataFrame:
        if not os.path.exists(model_file_path):
            data = pd.read_csv(train_file_path)
            self.train(data, save_path=model_file_path)

        model = self.load(model_file_path)
        predict_data = pd.read_csv(roberta_output_file_path)

        features = [
            "fake",
            "real",
            "negative",
            "neutral",
            "positive",
            "spam",
            "no_spam",
        ]
        X_pred = predict_data[features]
        y_pred = model.predict(X_pred)

        frame = pd.DataFrame(
            {
                "id_review": predict_data["id_review"],
                "id_room": predict_data["id_room"],
                "legit": y_pred,
            }
        )

        output_path = roberta_output_file_path.replace(
            ".csv", "_predicted.csv"
        )
        frame.to_csv(output_path, index=False)

        return frame


def roberta_classify_from_csv(file_path: str, api) -> None:
    df = pd.read_csv(file_path, nrows=150000)
    if "review" not in df.columns:
        raise ValueError("CSV file must contain a 'review' column.")

    if "id_review" not in df.columns:
        raise ValueError("CSV file must contain an 'id_review' column.")

    def_length = len(df)
    roberta_detector = RobertaOpenAIDetector(api_key=api)
    sentiment_analyzer = RobertaSentimentAnalyzer()
    otis_spam_detector = OtisAntiSpamAI(api_key=api)

    texts = df["review"].tolist()

    print("Processing fake/real detection in batch...")
    list_detections = roberta_detector.classify_batch(texts)
    print("Fake/real detection completed.")

    print("Processing sentiment analysis...")
    list_sentiments = []
    for i, text in enumerate(texts):
        print(f"Processing sentiment {i + 1}/{def_length}")
        list_sentiments.append(sentiment_analyzer.analyze(text))
    print("Sentiment analysis completed.")

    print("Processing spam detection in batch...")
    spam_results = otis_spam_detector.classify_batch(texts)
    print("Spam detection completed.")

    frame = pd.DataFrame(
        {
            "id_review": df["id_review"],
            "review": df["review"],
            "fake": [d["fake"] for d in list_detections],
            "real": [d["real"] for d in list_detections],
            "negative": [s["negative"] for s in list_sentiments],
            "neutral": [s["neutral"] for s in list_sentiments],
            "positive": [s["positive"] for s in list_sentiments],
            "spam": [s["spam"] for s in spam_results],
            "no_spam": [s["no_spam"] for s in spam_results],
        }
    )

    if "legit" in df.columns:
        frame["legit"] = [True if l == 1 else False for l in df["legit"]]

    if "id_room" in df.columns:
        frame["id_room"] = df["id_room"]

    output_path = file_path.replace(".csv", "_classified.csv")
    frame.to_csv(output_path, index=False)


def main():
    API_KEY = os.getenv("HF_API_KEY")
    if not API_KEY:
        raise ValueError("HF_API_KEY environment variable not set.")
    roberta_classify_from_csv("./backend/data/yelp_dataset.csv", api=API_KEY)
    downstream_model = DownstreamModel()
    downstream_model.train(
        data=pd.read_csv("./backend/data/yelp_dataset_classified.csv"),
        save_path="./backend/data/trained/downstream_rf_model.joblib",
        show_plot=True
    )

    roberta_classify_from_csv("./backend/data/reviews_en_clean.csv", api=API_KEY)

    downstream_model.predict_from_csv(
        model_file_path="./backend/data/trained/downstream_rf_model.joblib",
        roberta_output_file_path="./backend/data/reviews_en_clean_classified.csv",
        train_file_path="./backend/data/yelp_dataset_classified.csv"
    )


if __name__ == "__main__":
    main()
