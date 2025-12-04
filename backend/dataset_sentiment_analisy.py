import os
import pandas as pd
import torch
import matplotlib as mpl
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
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
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-classification",
            model="openai-community/roberta-base-openai-detector",
            device=device
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
                batch_size=batch_size
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
                        "bot": fake,
                        "no_bot": real,
                    }
                )
        return results


class RobertaSentimentAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self.model.eval()

    def analyze(self, text: str) -> dict:
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**encoded_input)
        scores = output.logits[0].detach().cpu().numpy()
        scores = softmax(scores)
        return {
            "negative": float(scores[0]),
            "neutral": float(scores[1]),
            "positive": float(scores[2]),
        }

    def analyze_batch(self, texts, batch_size: int = 32) -> list:
        texts = list(texts)
        total = len(texts)
        
        indexed_texts = sorted(enumerate(texts), key=lambda x: len(x[1]))
        sorted_texts = [t for i, t in indexed_texts]
        original_indices = [i for i, t in indexed_texts]
        
        final_results = [None] * total

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            print(f"Processing sentiment {end}/{total}")
            
            batch_texts = sorted_texts[start:end]
            batch_indices = original_indices[start:end]
            
            encoded_input = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self.device)
            
            with torch.no_grad():
                output = self.model(**encoded_input)
            
            scores_batch = output.logits.detach().cpu().numpy()
            scores_batch = softmax(scores_batch, axis=1)

            for i, scores in enumerate(scores_batch):
                original_idx = batch_indices[i]
                final_results[original_idx] = {
                    "negative": float(scores[0]),
                    "neutral": float(scores[1]),
                    "positive": float(scores[2]),
                }
        
        return final_results


class OtisAntiSpamAI:
    def __init__(self, api_key: str):
        device = 0 if torch.cuda.is_available() else -1
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )
        self.pipe = pipeline(
            "text-classification",
            model="Titeiiko/OTIS-Official-Spam-Model",
            device=device
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
                batch_size=batch_size
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
    
class TextBlobSubjectivityAnalyzer:
    def __init__(self):
        self.text_blob = TextBlob

    def analyze(self, text: str) -> float:
        blob = self.text_blob(text)
        return float(blob.sentiment.subjectivity)

    def analyze_batch(self, texts) -> list:
        results = []
        for text in texts:
            blob = self.text_blob(text)
            results.append(float(blob.sentiment.subjectivity))
        return results


def roberta_classify_from_csv(file_path: str, api) -> None:
    df = pd.read_csv(file_path)
    if "review" not in df.columns:
        raise ValueError("CSV file must contain a 'review' column.")

    if "id_review" not in df.columns:
        raise ValueError("CSV file must contain an 'id_review' column.")

    roberta_detector = RobertaOpenAIDetector()
    sentiment_analyzer = RobertaSentimentAnalyzer()
    otis_spam_detector = OtisAntiSpamAI(api_key=api)
    subjectivity_analyzer = TextBlobSubjectivityAnalyzer()

    texts = df["review"].tolist()

    print("Processing fake/real detection in batch...")
    list_detections = roberta_detector.classify_batch(texts)
    print("Fake/real detection completed.")

    print("Processing sentiment analysis in batch...")
    list_sentiments = sentiment_analyzer.analyze_batch(texts)
    print("Sentiment analysis completed.")

    print("Processing spam detection in batch...")
    spam_results = otis_spam_detector.classify_batch(texts)
    print("Spam detection completed.")
    print("Processing subjectivity analysis in batch...")
    list_subjectivities = subjectivity_analyzer.analyze_batch(texts)
    print("Subjectivity analysis completed.")


    print("Compiling results and saving to CSV...")

    frame = pd.DataFrame(
        {
            "id_review": df["id_review"],
            "review": df["review"],
            "bot": [d["bot"] for d in list_detections],
            "no_bot": [d["no_bot"] for d in list_detections],
            "negative": [s["negative"] for s in list_sentiments],
            "neutral": [s["neutral"] for s in list_sentiments],
            "positive": [s["positive"] for s in list_sentiments],
            "spam": [s["spam"] for s in spam_results],
            "no_spam": [s["no_spam"] for s in spam_results],
            "subjectivity": list_subjectivities,
        }
    )

    if "legit" in df.columns:
        frame["legit"] = [True if l == 1 else False for l in df["legit"]]

    if "rating" in df.columns:
        frame["rating"] = df["rating"]

    if "id_room" in df.columns:
        frame["id_room"] = df["id_room"]

    output_path = file_path.replace(".csv", "_classified.csv")
    frame.to_csv(output_path, index=False)


def main():
    API_KEY = os.getenv("HF_API_KEY")
    if not API_KEY:
        raise ValueError("HF_API_KEY environment variable not set.")
    roberta_classify_from_csv("./backend/data/yelp_dataset.csv", api=API_KEY)


if __name__ == "__main__":
    main()