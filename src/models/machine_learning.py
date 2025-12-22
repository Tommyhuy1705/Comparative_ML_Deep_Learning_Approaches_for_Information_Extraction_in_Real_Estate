import sklearn_crfsuite
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
        self.vectorizer_type = config.get('vectorizer', 'dict')
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
        
        if self.vectorizer_type == 'phobert':
            self.model = clf
        else:
            self.vectorizer = DictVectorizer(sparse=True)
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

class RelationExtractionModel:
    def __init__(self, model_type, config):
        self.vectorizer_type = config.get('vectorizer', 'dict')
        class_weight = config.get('class_weight', None)
        if model_type == 'svm':
            if self.vectorizer_type == 'phobert':
                clf = SVC(
                    kernel='rbf', 
                    C=config.get('C', 10.0),
                    probability=True,
                    class_weight=class_weight,
                    random_state=config.get('random_state', 42)
                )
            else:
                clf = SGDClassifier(
                    loss=config.get('loss', 'hinge'),
                    alpha=config.get('alpha', 1e-4),
                    class_weight=class_weight,
                    random_state=config.get('random_state', 42)
                )
        elif model_type == 'maxent':
            clf = LogisticRegression(
                solver=config.get('solver', 'lbfgs'),
                multi_class=config.get('multi_class', 'auto'),
                max_iter=config.get('max_iter', 1000),
                C=config.get('C', 1.0),
                random_state=config.get('random_state', 42),
                class_weight=class_weight
            )
        elif model_type == 'random_forest':
            clf = RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                criterion=config.get('criterion', 'gini'),
                max_depth=config.get('max_depth', None),
                min_samples_split=config.get('min_samples_split', 2),
                n_jobs=config.get('n_jobs', -1),
                random_state=config.get('random_state', 42),
                class_weight=class_weight
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if self.vectorizer_type == 'phobert':
            print(f"RE Model {model_type.upper()}: Using PhoBERT Vectors (Weighted)")
            if model_type == 'svm' or model_type == 'maxent':
                self.model = make_pipeline(StandardScaler(), clf)
            else:
                self.model = clf
        else:
            self.vectorizer = DictVectorizer(sparse=True)
            self.model = make_pipeline(self.vectorizer, clf)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)