import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline
from scipy.special import softmax
from sklearn.model_selection import RepeatedKFold
from huggingface_hub import InferenceClient


class RobertaOpenAIDetector:
    def __init__(self, api_key: str):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )
    
    def classify(self, text: str) -> dict:
        result = self.client.text_classification(   
            text,
            model="openai-community/roberta-base-openai-detector",
        )
        return {
            "fake": result[0]['score'],
            "real": result[1]['score'],
        }

class RobertaSentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def analyze(self, text: str) -> dict:
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return {
            "negative": scores[0],
            "neutral": scores[1],
            "positive": scores[2],
        }

class OtisAntiSpamAI:
    def __init__(self, api_key: str):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key)
    def classify(self, text: str) -> dict:
        pipe = pipeline("text-classification", model="Titeiiko/OTIS-Official-Spam-Model")
        x = pipe(text)[0]
        return {
                "spam":1 - x["score"], 
                "no_spam":x["score"]
        }
    


class DownstreamModel:
    def __init__(self, api_key: str):
        return
    
    def train(self, data: pd.DataFrame) -> GridSearchCV:
        features = ["fake", "real", "negative", "neutral", "positive", "spam", "no_spam"]
        X = data[features]


        rf_grid = RandomForestClassifier(random_state=42)
        gr_space = {
            'max_depth': [3,5,7,10],
            'n_estimators': [100, 200, 300, 400, 500],
            'max_features': [10, 20, 30 , 40],
            'min_samples_leaf': [1, 2, 4]
        }
        grid = GridSearchCV(rf_grid, gr_space, cv = 5, scoring='accuracy', verbose = 3)

        grid.fit(X, data['legit'])
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best cross-validation accuracy: {grid.best_score_}")
        self.save(grid, './backend/data/trained')

        return grid
    
    def save(self, model, path: str) -> None:
        best_rf_model = model.best_estimator_
        filename = path +'random_forest.joblib'
        joblib.dump(best_rf_model, filename)
        print(f"Model saved as '{filename}'")
    
    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path '{path}' does not exist.")
        return joblib.load(path)
        
    
    def predict_from_csv(self, model_file_path: str, roberta_output_file_path: str, train_file_path: str) -> None:
        if not os.path.exists(model_file_path):
            data = pd.read_csv(train_file_path)
            self.train(data)

        model = self.load(model_file_path)
        predict_data = pd.read_csv(roberta_output_file_path)
        
        frame = pd.DataFrame(columns=["id_review","id_room", "legit"])
        for i in range(len(predict_data)):
            frame = frame.append({
                "id_review": predict_data.loc[i, 'id_review'],
                "id_room": predict_data.loc[i, 'id_room'],
                "legit": model.predict([[
                    predict_data.loc[i, 'fake'],
                    predict_data.loc[i, 'real'],
                    predict_data.loc[i, 'negative'],
                    predict_data.loc[i, 'neutral'],
                    predict_data.loc[i, 'positive'],
                    predict_data.loc[i, 'spam'],
                    predict_data.loc[i, 'no_spam']
                ]])[0]
            }, ignore_index=True)

        frame.to_csv(roberta_output_file_path.replace(".csv", "_predicted.csv"), index=False)

        return frame
      
    
       
def classify_from_csv(file_path: str, api) -> None:
    df = pd.read_csv(file_path)
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain a 'text' column.")
    

    def_length = len(df)
    roberta_detector = RobertaOpenAIDetector(api_key=api)
    sentiment_analyzer = RobertaSentimentAnalyzer()
    otis_spam_detector = OtisAntiSpamAI(api_key=api)
    list_detections = []
    list_sentiments = []
    list_spams = [] 

    for i, row in df.iterrows():
        print(f"Processing review {i+1}/{def_length}")
        review = row['review']
        list_detections.append(roberta_detector.classify(review))
        list_sentiments.append(sentiment_analyzer.analyze(review))
        list_spams.append(otis_spam_detector.classify(review))

    frame = pd.DataFrame(columns=["id_review","review","id_room", "fake", "real", "negative", "neutral", "positive", "spam", "no_spam"])
    for i in range(len(df)):
        frame = frame.append({
            "id_review": df.loc[i, 'id_review'],
            "review": df.loc[i, 'review'],
            "id_room": df.loc[i, 'id_room'],
            "fake": list_detections[i]['fake'],
            "real": list_detections[i]['real'],
            "negative": list_sentiments[i]['negative'],
            "neutral": list_sentiments[i]['neutral'],
            "positive": list_sentiments[i]['positive'],
            "spam": list_spams[i]['spam'],
            "no_spam": list_spams[i]['no_spam'],
            "legit": True
        }, ignore_index=True)

    output_path = file_path.replace(".csv", "_classified.csv")
    frame.to_csv(output_path, index=False)



def main():
    API_KEY = os.getenv("HF_API_KEY")
    if not API_KEY:
        raise ValueError("HF_API_KEY environment variable not set.")
    classify_from_csv("./backend/data/reviews_en_clean.csv", api=API_KEY)

if __name__ == "__main__":
    main()
