import pandas as pd
import numpy as np
import warnings
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class YelpFraudDetector:
    def __init__(self, file_path):
        self.file_path = file_path
        self.target_col = 'legit'
        
        self.feature_cols = [
            'bot', 'no_bot', 'negative', 'neutral', 'positive', 
            'spam', 'no_spam', 'subjectivity', 'rating', 'char_len', 
            'word_count', 'money_count', 'money_ratio_sent', 
            'pos_ADJ_count', 'pos_ADJ_ratio_sent', 'pos_ADP_count', 
            'pos_ADP_ratio_sent', 'pos_ADV_count', 'pos_ADV_ratio_sent', 
            'pos_AUX_count', 'pos_PRON_count', 'pos_PRON_ratio_sent', 
            'pos_PROPN_count', 'pos_VERB_count', 'verb_count', 
            'verb_ratio_sent', 'verb_tense_Past_count', 
            'verb_tense_Past_ratio_sent', 'verb_tense_Pres_count', 
            'verb_tense_Pres_ratio_sent'
        ]
        self.best_model = None
        self.best_features_names = []

    def load_data(self):
        print(f"\n--- Loading Data from {self.file_path} ---")
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise ValueError(f"CRITICAL ERROR: File '{self.file_path}' not found.")

        print(f"Total rows found: {len(df)}")
        
        # 1. Fill missing numeric features with 0
        df[self.feature_cols] = df[self.feature_cols].fillna(0)

        # 2. Drop rows where Target is missing
        df_clean = df.dropna(subset=[self.target_col])
        
        X = df_clean[self.feature_cols]
        y = df_clean[self.target_col]
        
        print(f"Final Training Size: {len(df_clean)} rows")
        print(f"Target Distribution:\n{y.value_counts()}")

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train(self, X_train, y_train, balance_method='original', model_type='logreg'):
        start_time = time.time()
        print(f"\n>>> TRAINING START: Model=[{model_type.upper()}] | Strategy=[{balance_method.upper()}]")
        
        steps = []
        
        # 1. Sampling
        if balance_method == 'undersample':
            steps.append(('sampler', RandomUnderSampler(sampling_strategy=0.5, random_state=42)))
        elif balance_method == 'oversample':
            steps.append(('sampler', RandomOverSampler(sampling_strategy=0.5, random_state=42)))
        
        # 2. Preprocessing
        steps.append(('scaler', PowerTransformer())) # PowerTransformer is better for skewed data
        steps.append(('selector', SelectKBest(score_func=f_classif)))

        param_dist = []
        if model_type == 'logreg':
            cw = 'balanced' if balance_method == 'original' else None
            steps.append(('clf', LogisticRegression(
                solver='saga', max_iter=8000, tol=1e-3, class_weight=cw, random_state=42
            )))
            
            common_k = list(range(3, len(self.feature_cols) + 1))
            common_C = np.logspace(-3, 3, 50) 

            param_dist = [
                {'selector__k': common_k, 'clf__C': common_C, 'clf__penalty': ['l1', 'l2']},
                {'selector__k': common_k, 'clf__C': common_C, 'clf__penalty': ['elasticnet'], 'clf__l1_ratio': np.linspace(0.1, 0.9, 10)}
            ]
            n_iter_search = 40 
            
        elif model_type == 'nn':
            steps.append(('clf', MLPClassifier(
                max_iter=2000, early_stopping=True, n_iter_no_change=10, random_state=42, hidden_layer_sizes=1
            )))
            
            param_dist = {
                'selector__k': list(range(5, len(self.feature_cols) + 1)),
                'clf__hidden_layer_sizes': [(), (32,), (64,), (100, 50)], 
                'clf__activation': ['relu', 'tanh'],
                'clf__alpha': [0.0001, 0.001, 0.01],
                'clf__learning_rate_init': [0.001, 0.01]
            }
            n_iter_search = 20 

        pipeline = ImbPipeline(steps)

        # 4. Randomized Search
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=cv,
            scoring='f1_macro',
            verbose=3,
            random_state=42,
            n_jobs=-1
        )

        try:
            search.fit(X_train, y_train)
        except Exception as e:
            print(f"TRAINING FAILED: {e}")
            return None, 0

        duration = time.time() - start_time
        self.best_model = search.best_estimator_
        

        selector = self.best_model.named_steps['selector']
        mask = selector.get_support()
        self.best_features_names = [f for f, kept in zip(self.feature_cols, mask) if kept]
        
        print(f"Done in {duration:.1f}s. Best CV F1: {search.best_score_:.4f}")
        return search.best_params_, search.best_score_

    def evaluate(self, X_test, y_test):
        if self.best_model is None: return 0

        y_pred = self.best_model.predict(X_test)
        score = f1_score(y_test, y_pred, average='macro')
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"-> Features Used: {len(self.best_features_names)}")
        return score

def main():
    csv_file = './backend/data/yelp_reviews_merged_features.csv'
    detector = YelpFraudDetector(csv_file)
    
    try:
        X_train, X_test, y_train, y_test = detector.load_data()
    except ValueError as e:
        print(e)
        return

    models = ['logreg', 'nn']
    strategies = ['original', 'undersample', 'oversample']

    results = []
    global_best_score = 0
    global_best_info = {}

    for model in models:
        for strategy in strategies:
            best_params, _ = detector.train(X_train, y_train, 
                                                 balance_method=strategy, 
                                                 model_type=model)
            
            if best_params:
                test_score = detector.evaluate(X_test, y_test)
                
                results.append({
                    'Model': model.upper(),
                    'Strategy': strategy.upper(),
                    'Test_F1': test_score
                })
                
                if test_score > global_best_score:
                    global_best_score = test_score
                    global_best_info = {
                        'Model': model.upper(),
                        'Strategy': strategy.upper(),
                        'Params': best_params,
                        'Features': detector.best_features_names
                    }
            print("-" * 60)

    if results:
        print("\n" + "="*60)
        print(" FINAL LEADERBOARD ")
        print("="*60)
        print(pd.DataFrame(results).sort_values(by='Test_F1', ascending=False).to_string(index=False))
        
        print("\n" + "*"*60)
        print(f"CHAMPION: {global_best_info['Model']} ({global_best_info['Strategy']})")
        print(f"F1 Score: {global_best_score:.4f}")
        print(f"\n🏆 FEATURES ({len(global_best_info['Features'])}):")
        print(global_best_info['Features'])
        print("*"*60)

if __name__ == "__main__":
    main()