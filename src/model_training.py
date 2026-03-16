from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class ResumeClassifier:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the classification model"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'svm':
            return SVC(kernel='linear', random_state=42)
        elif self.model_type == 'naive_bayes':
            return MultinomialNB()
        else:
            raise ValueError("Invalid model type")

    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('models/confusion_matrix.png')
        plt.close()

        return y_pred

    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)