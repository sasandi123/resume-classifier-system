# AI-Based Resume Classification System

An intelligent NLP-powered system that automatically classifies resumes into job categories.

## Features
- PDF and DOCX resume parsing
- Multi-class job category classification
- Skill extraction
- Contact information extraction
- Web-based interface
- High accuracy with machine learning

## Tech Stack
- Python 3.8+
- scikit-learn
- NLTK & spaCy
- Flask
- TF-IDF Vectorization
- Random Forest Classifier

## Installation
```bash
git clone https://github.com/yourusername/resume-classification-system.git
cd resume-classification-system
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

1. Train the model:
```bash
python train.py
```

2. Run the web app:
```bash
cd app
python app.py
```

3. Open browser to `http://localhost:5000`

## Project Structure
```
resume-classification-system/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   └── model_training.py
├── app/
│   ├── app.py
│   └── templates/
├── notebooks/
├── requirements.txt
└── README.md
```

## Dataset
Download resume dataset from [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

## Results
- Accuracy: ~85-90%
- Supports multiple job categories

## Future Improvements
- Deep learning models (BERT)
- Resume ranking system
- Job matching recommendations

## Author
Your Name - AI & Data Science Student