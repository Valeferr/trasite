import os
import joblib
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
    pass


class RobertaOpenAIDetector:
    def __init__(self, api_key: str):
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

    def classify_batch(self, texts):
        outputs = self.pipe(
            list(texts),
            truncation=True,
            max_length=512,
        )
        results = []
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
    def train(self, data: pd.DataFrame, save_path: str, show_plot: bool = False, plot_path: str = None) -> RandomizedSearchCV:
        np.random.seed(42)
        ax = data.hist(column="legit")
        if plot_path is not None:
            plt.savefig(plot_path)
        if show_plot:
            plt.show()
        plt.close()
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

        rows = []
        for i in range(len(predict_data)):
            rows.append(
                {
                    "id_review": predict_data.loc[i, "id_review"],
                    "id_room": predict_data.loc[i, "id_room"],
                    "legit": model.predict(
                        [
                            [
                                predict_data.loc[i, "fake"],
                                predict_data.loc[i, "real"],
                                predict_data.loc[i, "negative"],
                                predict_data.loc[i, "neutral"],
                                predict_data.loc[i, "positive"],
                                predict_data.loc[i, "spam"],
                                predict_data.loc[i, "no_spam"],
                            ]
                        ]
                    )[0],
                }
            )

        frame = pd.DataFrame(rows, columns=["id_review", "id_room", "legit"])

        output_path = roberta_output_file_path.replace(
            ".csv", "_predicted.csv"
        )
        frame.to_csv(output_path, index=False)

        return frame


def roberta_classify_from_csv(file_path: str, api) -> None:
    df = pd.read_csv(file_path, nrows=100000)
    if "review" not in df.columns:
        raise ValueError("CSV file must contain a 'review' column.")

    if "id_review" not in df.columns:
        raise ValueError("CSV file must contain an 'id_review' column.")

    if "legit" not in df.columns:
        raise ValueError("CSV file must contain a 'legit' column.")

    def_length = len(df)
    roberta_detector = RobertaOpenAIDetector(api_key=api)
    sentiment_analyzer = RobertaSentimentAnalyzer()
    otis_spam_detector = OtisAntiSpamAI(api_key=api)

    list_detections = []
    list_sentiments = []

    for i, row in df.iterrows():
        print(f"Processing review {i + 1}/{def_length}")
        review = row["review"]
        list_detections.append(roberta_detector.classify(review))
        list_sentiments.append(sentiment_analyzer.analyze(review))

    print("Processing spam detection in batch...")
    spam_results = otis_spam_detector.classify_batch(df["review"].tolist())
    print("Spam detection completed.")
    rows = []
    for i in range(len(df)):
        rows.append(
            {
                "id_review": df.loc[i, "id_review"],
                "review": df.loc[i, "review"],
                "fake": list_detections[i]["fake"],
                "real": list_detections[i]["real"],
                "negative": list_sentiments[i]["negative"],
                "neutral": list_sentiments[i]["neutral"],
                "positive": list_sentiments[i]["positive"],
                "spam": spam_results[i]["spam"],
                "no_spam": spam_results[i]["no_spam"],
                "legit": True if int(df.loc[i, "legit"]) == 1 else False,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "id_review",
            "review",
            "fake",
            "real",
            "negative",
            "neutral",
            "positive",
            "spam",
            "no_spam",
            "legit",
        ],
    )

    output_path = file_path.replace(".csv", "_classified.csv")
    frame.to_csv(output_path, index=False)


def main():
    API_KEY = os.getenv("HF_API_KEY")
    if not API_KEY:
        raise ValueError("HF_API_KEY environment variable not set.")
    roberta_classify_from_csv("./backend/data/yelp_dataset.csv", api=API_KEY)

    downstream_model= DownstreamModel()
    downstream_model.train(
        data=pd.read_csv("./backend/data/yelp_dataset_classified.csv"),
        save_path="./backend/data/trained/downstream_rf_model.joblib"
    )

    roberta_classify_from_csv("./backend/data/reviews_en_clean.csv", api=API_KEY)

    downstream_model.predict_from_csv( 
        model_file_path="./backend/data/trained/downstream_rf_model.joblib",
        roberta_output_file_path="./backend/data/reviews_en_clean_classified.csv",
        train_file_path="./backend/data/yelp_dataset_classified.csv"
    )


if __name__ == "__main__":
    main()
