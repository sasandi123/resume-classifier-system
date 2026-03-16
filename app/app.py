from flask import Flask, render_template, request, jsonify
import joblib
import sys
import os

sys.path.append('..')

from src.data_preprocessing import ResumeParser

app = Flask(__name__)

# Load model, vectorizer, and categories
try:
    model = joblib.load('../models/resume_classifier.pkl')
    tfidf = joblib.load('../models/tfidf_vectorizer.pkl')
    categories = joblib.load('../models/categories.pkl')
    parser = ResumeParser()
    print(" Models loaded successfully!")
    print(f" Categories available ({len(categories)}): {categories}")
except Exception as e:
    print(f" Error loading models: {e}")
    print(" Please train the model first by running: python train.py")
    model = None
    tfidf = None
    categories = []
    parser = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not tfidf or not parser:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        # Check if file is present
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['resume']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Extract text based on file type
        text = None
        temp_path = None

        try:
            if file.filename.endswith('.pdf'):
                # Save temporarily
                temp_path = f'temp_{os.urandom(8).hex()}_{file.filename}'
                file.save(temp_path)
                text = parser.extract_text_from_pdf(temp_path)

            elif file.filename.endswith('.docx'):
                temp_path = f'temp_{os.urandom(8).hex()}_{file.filename}'
                file.save(temp_path)
                text = parser.extract_text_from_docx(temp_path)

            elif file.filename.endswith('.txt'):
                text = file.read().decode('utf-8')

            else:
                return jsonify({'error': 'Unsupported file format. Please use PDF, DOCX, or TXT'}), 400

        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

        # Validate extracted text
        if not text or len(text.strip()) < 50:
            return jsonify({'error': 'Could not extract enough text from resume. Please check the file.'}), 400

        print(f"\n{'=' * 60}")
        print(f"Processing resume: {file.filename}")
        print(f"Extracted text length: {len(text)} characters")

        # Preprocess with SMART extraction (focuses on relevant sections)
        print("Extracting focused content (Skills, Education, Summary)...")
        cleaned_text = parser.clean_text(text)
        processed_text = parser.preprocess_text(cleaned_text)

        print(f"Processed text length: {len(processed_text)} characters")

        if len(processed_text.strip()) < 20:
            return jsonify(
                {'error': 'Resume content too short after processing. Please ensure resume has sufficient text.'}), 400

        # Extract features
        print("Extracting TF-IDF features...")
        features = tfidf.transform([processed_text])

        # Predict
        print("Making prediction...")
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100

        # Get top 3 predictions
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_predictions = [
            {
                'category': model.classes_[idx],
                'confidence': f'{probabilities[idx] * 100:.2f}%',
                'probability': float(probabilities[idx])
            }
            for idx in top_3_indices
        ]

        print(f"Predicted: {prediction} ({confidence:.2f}%)")
        print(f"Top 3: {[p['category'] for p in top_3_predictions]}")

        # Extract additional info using SMART extraction
        print("Extracting contact info and skills...")
        email = parser.extract_email(text)  # Avoids references section
        phone = parser.extract_phone(text)  # International support
        skills = parser.extract_skills(text)

        print(f"Email: {email}")
        print(f"Phone: {phone}")
        print(f"Skills found: {len(skills)}")
        print(f"{'=' * 60}\n")

        # Build result
        result = {
            'category': prediction,
            'confidence': f'{confidence:.2f}%',
            'top_predictions': top_3_predictions,
            'skills': skills[:15],  # Top 15 skills
            'total_skills': len(skills),
            'email': email,
            'phone': phone,
            'resume_length': len(text),
            'word_count': len(text.split()),
            'processed_length': len(processed_text)
        }

        return jsonify(result)

    except Exception as e:
        print(f" Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/categories', methods=['GET'])
def get_categories():
    """Return all available categories"""
    try:
        return jsonify({'categories': categories, 'total': len(categories)})
    except:
        return jsonify({'error': 'Categories not loaded'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'ok' if model else 'error',
        'model_loaded': model is not None,
        'tfidf_loaded': tfidf is not None,
        'parser_loaded': parser is not None,
        'categories_count': len(categories)
    }
    return jsonify(status)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" Starting Resume Classification System")
    print("=" * 60)
    print(f"Model status: {' Loaded' if model else ' Not loaded'}")
    print(f"Categories: {len(categories)}")
    print(f"Server: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5000)