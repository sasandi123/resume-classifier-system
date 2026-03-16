import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import ResumeParser
from src.feature_extraction import FeatureExtractor
from src.model_training import ResumeClassifier


def analyze_sample_resumes(df, category, n_samples=3):
    """Analyze sample resumes from a category to understand what the model sees"""
    samples = df[df['Category'] == category].head(n_samples)

    print(f"\n{'=' * 80}")
    print(f"SAMPLE RESUMES FROM: {category}")
    print(f"{'=' * 80}")

    for idx, row in samples.iterrows():
        print(f"\n--- Sample {idx + 1} ---")
        print(f"Original (first 500 chars):")
        print(row['Resume_str'][:500])
        print(f"\nProcessed (first 300 chars):")
        print(row['Processed_Resume'][:300])
        print("-" * 80)


def main():
    # Initialize components
    print("Initializing Resume Parser and Feature Extractor...")
    parser = ResumeParser()
    feature_extractor = FeatureExtractor()

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    df = pd.read_csv('data/raw/Resume.csv')

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())

    # Remove missing values
    df = df.dropna(subset=['Resume_str', 'Category'])
    print(f"\nDataset after removing NaN: {df.shape[0]} resumes")

    # Category distribution
    print(f"\n{'=' * 80}")
    print(f"CATEGORY DISTRIBUTION")
    print(f"{'=' * 80}")
    print(f"Total categories: {df['Category'].nunique()}")
    print("\nResumes per category:")
    print(df['Category'].value_counts())

    # Preprocess resumes with SMART content extraction
    print(f"\n{'=' * 80}")
    print("PREPROCESSING RESUMES")
    print("=" * 80)
    print("Extracting focused content (Summary, Skills, Education, Certifications)...")
    print("Ignoring misleading content (Project names, References, Extracurricular)...")

    df['Cleaned_Resume'] = df['Resume_str'].apply(parser.clean_text)
    df['Processed_Resume'] = df['Cleaned_Resume'].apply(parser.preprocess_text)

    # Show sample of what the model actually sees
    print("\n" + "=" * 80)
    print("WHAT THE MODEL SEES (Sample)")
    print("=" * 80)
    sample_idx = 0
    print(f"Original Resume (first 800 chars):")
    print(df['Resume_str'].iloc[sample_idx][:800])
    print(f"\n{'~' * 80}")
    print(f"Processed Resume (what model uses for classification):")
    print(df['Processed_Resume'].iloc[sample_idx][:600])

    # Save processed sample for inspection
    sample_df = df[['Category', 'Resume_str', 'Processed_Resume']].head(20)
    sample_df.to_csv('data/processed/sample_processed_resumes.csv', index=False)
    print(f"\n✓ Saved 20 sample processed resumes to: data/processed/sample_processed_resumes.csv")

    # Extract features using TF-IDF
    print(f"\n{'=' * 80}")
    print("EXTRACTING TF-IDF FEATURES")
    print("=" * 80)
    X = feature_extractor.fit_tfidf(df['Processed_Resume'])
    y = df['Category']

    print(f"Feature matrix shape: {X.shape}")
    print(f"  - Resumes: {X.shape[0]}")
    print(f"  - Features (words/phrases): {X.shape[1]}")
    print(f"Number of unique categories: {y.nunique()}")

    # Show top TF-IDF features
    feature_names = feature_extractor.get_feature_names()
    print(f"\nSample TF-IDF features (random 20):")
    sample_features = np.random.choice(feature_names, 20, replace=False)
    for i, feat in enumerate(sample_features, 1):
        print(f"  {i:2d}. {feat}")

    # Split data (80% train, 20% test)
    print(f"\n{'=' * 80}")
    print("SPLITTING DATASET")
    print("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} resumes ({X_train.shape[0] / len(df) * 100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} resumes ({X_test.shape[0] / len(df) * 100:.1f}%)")

    # Train Random Forest Classifier
    print(f"\n{'=' * 80}")
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 80)
    print("Model: Random Forest with 200 trees")
    print("Training in progress...")

    classifier = ResumeClassifier(model_type='random_forest')
    classifier.train(X_train, y_train)

    print("✓ Training complete!")

    # Evaluate model
    print(f"\n{'=' * 80}")
    print("MODEL EVALUATION")
    print("=" * 80)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\n Detailed Classification Report:")
    print("=" * 80)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    print("\n Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classifier.model.classes_,
                yticklabels=classifier.model.classes_)
    plt.title('Confusion Matrix - Resume Classification', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved confusion matrix to: models/confusion_matrix.png")

    # Feature importance analysis
    print(f"\n{'=' * 80}")
    print("TOP 50 MOST IMPORTANT FEATURES FOR CLASSIFICATION")
    print("=" * 80)

    if hasattr(classifier.model, 'feature_importances_'):
        importances = classifier.model.feature_importances_
        indices = np.argsort(importances)[::-1][:50]

        print("\nThese words/phrases have the highest impact on classification:")
        print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12}")
        print("-" * 50)

        for rank, idx in enumerate(indices, 1):
            print(f"{rank:<6} {feature_names[idx]:<30} {importances[idx]:.6f}")

    # Analyze misclassifications
    print(f"\n{'=' * 80}")
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 80)

    misclassified = y_test != y_pred
    if misclassified.sum() > 0:
        print(f"Total misclassifications: {misclassified.sum()} out of {len(y_test)}")
        print(f"Error rate: {misclassified.sum() / len(y_test) * 100:.2f}%")

        # Show some misclassifications
        misclass_df = pd.DataFrame({
            'True': y_test[misclassified],
            'Predicted': y_pred[misclassified]
        })

        print("\nMost common misclassification pairs:")
        misclass_pairs = misclass_df.groupby(['True', 'Predicted']).size().sort_values(ascending=False).head(10)
        for (true_label, pred_label), count in misclass_pairs.items():
            print(f"  {true_label:25s} → {pred_label:25s} : {count} times")
    else:
        print(" Perfect classification! No misclassifications!")

    # Save model artifacts
    print(f"\n{'=' * 80}")
    print("SAVING MODEL ARTIFACTS")
    print("=" * 80)

    classifier.save_model('models/resume_classifier.pkl')
    joblib.dump(feature_extractor.tfidf, 'models/tfidf_vectorizer.pkl')

    categories = sorted(df['Category'].unique().tolist())
    joblib.dump(categories, 'models/categories.pkl')

    print("✓ Saved successfully:")
    print("  - models/resume_classifier.pkl")
    print("  - models/tfidf_vectorizer.pkl")
    print("  - models/categories.pkl")
    print("  - models/confusion_matrix.png")

    # Test with real-world examples
    print(f"\n{'=' * 80}")
    print("TESTING WITH SAMPLE RESUMES")
    print("=" * 80)

    test_samples = [
        {
            'name': 'IT/Data Science Resume',
            'text': """
            AI and Data Science undergraduate student
            Education: BSc Artificial Intelligence and Data Science
            Skills: Python, Java, Machine Learning, Deep Learning, TensorFlow, 
            PyTorch, Firebase, MySQL, Flutter, Git
            Certifications: Oracle Cloud AI Certification, Java Programming
            """
        },
        {
            'name': 'Healthcare Professional',
            'text': """
            Registered Nurse with 5 years experience
            Education: Bachelor of Nursing
            Skills: Patient care, Medical records, Clinical assessment,
            Emergency response, Healthcare management
            Certifications: BLS, ACLS, Critical Care Nursing
            """
        },
        {
            'name': 'Finance Professional',
            'text': """
            Financial Analyst with CFA certification
            Education: MBA in Finance
            Skills: Financial modeling, Excel, SAP, Budgeting, Forecasting,
            Financial analysis, Investment analysis
            Certifications: CFA Level 2, Financial Risk Manager
            """
        }
    ]

    for sample in test_samples:
        print(f"\n--- {sample['name']} ---")
        cleaned = parser.clean_text(sample['text'])
        processed = parser.preprocess_text(cleaned)
        features = feature_extractor.tfidf.transform([processed])

        prediction = classifier.model.predict(features)[0]
        probabilities = classifier.model.predict_proba(features)[0]
        confidence = max(probabilities) * 100

        # Top 3 predictions
        top_3_idx = np.argsort(probabilities)[::-1][:3]

        print(f"Predicted: {prediction}")
        print(f"Confidence: {confidence:.2f}%")
        print("Top 3 predictions:")
        for i, idx in enumerate(top_3_idx, 1):
            print(f"  {i}. {classifier.model.classes_[idx]:30s} - {probabilities[idx] * 100:6.2f}%")

    # Final summary
    print(f"\n{'=' * 80}")
    print(" TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n Model Performance Summary:")
    print(f"   - Accuracy: {accuracy * 100:.2f}%")
    print(f"   - Categories: {len(categories)}")
    print(f"   - Training samples: {X_train.shape[0]}")
    print(f"   - Test samples: {X_test.shape[0]}")

    print(f"\n Next Steps:")
    print("   1. Review confusion matrix: models/confusion_matrix.png")
    print("   2. Check sample processed resumes: data/processed/sample_processed_resumes.csv")
    print("   3. Test with real resumes in the web app")
    print("   4. Run: cd app && python app.py")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()