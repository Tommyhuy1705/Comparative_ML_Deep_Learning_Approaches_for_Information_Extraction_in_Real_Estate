import sklearn_crfsuite
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

class CRFModel:
    def __init__(self, config):
        self.model = sklearn_crfsuite.CRF(
            algorithm=config['algorithm'],
            c1=config['c1'],
            c2=config['c2'],
            max_iterations=config['max_iterations'],
            all_possible_transitions=config['all_possible_transitions']
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)

# SVM và MaxEnt không xử lý chuỗi trực tiếp -> làm phẳng
class FlatModelWrapper:
    def __init__(self, model_type, config):
        self.vectorizer = DictVectorizer(sparse=True)
        if model_type == 'svm':
            clf = SGDClassifier(
                loss=config['loss'], 
                penalty=config['penalty'],
                alpha=config['alpha'],
                max_iter=config['max_iter'],
                random_state=config['random_state']
            )
        elif model_type == 'maxent':
            clf = LogisticRegression(
                solver=config['solver'],
                multi_class=config['multi_class'],
                max_iter=config['max_iter'],
                C=config.get('C', 1.0),
                random_state=config['random_state']
            )
        
        self.model = make_pipeline(self.vectorizer, clf)

    def _flatten(self, X):
        return [item for sublist in X for item in sublist]

    def train(self, X_train, y_train):
        X_flat = self._flatten(X_train)
        y_flat = self._flatten(y_train)
        self.model.fit(X_flat, y_flat)

    def predict(self, X_test):
        X_flat = self._flatten(X_test)
        y_pred_flat = self.model.predict(X_flat)
        
        y_pred_grouped = []
        idx = 0
        for sent in X_test:
            length = len(sent)
            y_pred_grouped.append(y_pred_flat[idx : idx + length].tolist())
            idx += length
        return y_pred_grouped
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)