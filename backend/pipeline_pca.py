import pandas as pd
import numpy as np
import warnings
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import classification_report, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class YelpFraudDetectorPCA:
    def __init__(self, file_path):
        self.file_path = file_path
        self.target_col = 'legit'
        
        # Feature PCA: pc1 -> pc175
        self.feature_cols = [f'PC{i}' for i in range(1, 176)]
        
        self.best_model = None
        self.n_features_in = len(self.feature_cols)

    def load_data(self):
        print(f"\n--- Loading Data from {self.file_path} ---")
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise ValueError(f"CRITICAL ERROR: File '{self.file_path}' not found.")

        print(f"Total rows found: {len(df)}")
        
        # Fill missing numeric features with 0
        df[self.feature_cols] = df[self.feature_cols].fillna(0)

        if self.target_col not in df.columns:
             raise ValueError(f"Target column '{self.target_col}' not found in CSV.")

        df_clean = df.dropna(subset=[self.target_col])
        
        X = df_clean[self.feature_cols]
        y = df_clean[self.target_col]
        
        print(f"Final Training Size: {len(df_clean)} rows")
        print(f"Target Distribution:\n{y.value_counts()}")

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    @staticmethod
    def _minority_majority_ratio(y):
        value_counts = pd.Series(y).value_counts(dropna=False)
        if len(value_counts) < 2:
            return None
        majority = value_counts.max()
        minority = value_counts.min()
        if majority == 0:
            return None
        return float(minority) / float(majority)

    def train(self, X_train, y_train, balance_method='original', model_type='logreg'):
        start_time = time.time()
        print(f"\n>>> TRAINING START: Model=[{model_type.upper()}] | Strategy=[{balance_method.upper()}]")
        
        steps = []
        
        majority_to_minority = 2.0
        desired_ratio = 1.0 / majority_to_minority
        n_classes = pd.Series(y_train).nunique(dropna=False)
        current_ratio = self._minority_majority_ratio(y_train)

        if balance_method in {'undersample', 'oversample'}:
            if n_classes > 2:
                strategy = 'auto'
                if balance_method == 'undersample':
                    steps.append(('sampler', RandomUnderSampler(sampling_strategy=strategy, random_state=42)))
                else:
                    steps.append(('sampler', RandomOverSampler(sampling_strategy=strategy, random_state=42)))
            else:
                if current_ratio is None:
                    print("Sampling skipped: not enough class variation")
                elif current_ratio >= desired_ratio:
                    print(f"Sampling skipped: current ratio {current_ratio:.3f} >= target {desired_ratio:.3f}")
                else:
                    if balance_method == 'undersample':
                        steps.append(('sampler', RandomUnderSampler(sampling_strategy=desired_ratio, random_state=42)))
                    else:
                        steps.append(('sampler', RandomOverSampler(sampling_strategy=desired_ratio, random_state=42)))
        
        steps.append(('scaler', PowerTransformer()))
        
        param_dist = []
        n_iter_search = 10 

        if model_type == 'logreg':
            cw = 'balanced' if balance_method == 'original' else None
            steps.append(('clf', LogisticRegression(
                solver='saga', 
                max_iter=8000, 
                tol=1e-3, 
                class_weight=cw, 
                random_state=42
            )))
            
            common_C = np.logspace(-3, 3, 50) 
            param_dist = [
                {'clf__C': common_C, 'clf__penalty': ['l1', 'l2']},
                {'clf__C': common_C, 'clf__penalty': ['elasticnet'], 'clf__l1_ratio': np.linspace(0.1, 0.9, 10)}
            ]
            
        elif model_type == 'nn':
            steps.append(('clf', MLPClassifier(
                hidden_layer_sizes=(128,),
                max_iter=1000, 
                early_stopping=True, 
                n_iter_no_change=10, 
                random_state=42,
            )))
            
            param_dist = {
                'clf__hidden_layer_sizes': [(32,), (64,), (128,), (175,)], 
                'clf__activation': ['relu', 'tanh'],
                'clf__alpha': [0.0001, 0.001, 0.01],
                'clf__learning_rate_init': [0.001, 0.01]
            }

        pipeline = ImbPipeline(steps)

        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

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
        
        print("\n" + "-"*30)
        print(" DETAILED FIT RESULTS (Sorted by F1 Macro) ")
        print("-"*30)
        
        results_df = pd.DataFrame(search.cv_results_)

        keep_cols = ['params', 'mean_test_score', 'mean_fit_time', 'rank_test_score']
        
        log_df = results_df[keep_cols].sort_values(by='mean_test_score', ascending=False)
        

        log_df = log_df.rename(columns={'mean_test_score': 'F1_Macro_Score', 'mean_fit_time': 'Time(s)'})
        
        print(log_df.to_string(index=False))
        print("-"*30 + "\n")

        print(f"Done in {duration:.1f}s. Best Validation F1: {search.best_score_:.4f}")
        return search.best_params_, search.best_score_

    def evaluate(self, X_test, y_test):
        if self.best_model is None: return 0

        y_pred = self.best_model.predict(X_test)
        score = f1_score(y_test, y_pred, average='macro')
        
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"-> Input Features: {self.n_features_in} (PCA Components)")
        return score

def main():
    csv_file = './backend/data/yelp_reviews_ml_ready_pca.csv'
    
    print(f"Initializing detector with file: {csv_file}")
    detector = YelpFraudDetectorPCA(csv_file)
    
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
                        'Params': best_params
                    }
            print("-" * 60)

    if results:
        print("\n" + "="*60)
        print(" FINAL LEADERBOARD (NO CV - SINGLE SPLIT) ")
        print("="*60)
        print(pd.DataFrame(results).sort_values(by='Test_F1', ascending=False).to_string(index=False))
        
        print("\n" + "*"*60)
        print(f"CHAMPION: {global_best_info['Model']} ({global_best_info['Strategy']})")
        print(f"F1 Score: {global_best_score:.4f}")
        print(f"Best Params: {global_best_info['Params']}")
        print("*"*60)

if __name__ == "__main__":
    main()