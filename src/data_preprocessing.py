import re
import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download NLTK data - UPDATED WITH ALL REQUIRED PACKAGES
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    print("Some NLTK downloads failed, but continuing...")


class ResumeParser:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def extract_text_from_docx(self, docx_path):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""

    def extract_resume_sections(self, text):
        """
        Extract different sections from resume intelligently.
        This is the KEY to proper classification - focus on relevant sections.
        """
        sections = {
            'summary': '',
            'objective': '',
            'skills': '',
            'education': '',
            'certifications': '',
            'experience': '',
            'projects': '',
            'references': '',
            'achievements': '',
            'extracurricular': ''
        }

        # Split text into lines for better section detection
        lines = text.split('\n')
        current_section = None

        # Section headers to detect
        section_patterns = {
            'summary': r'\b(summary|profile|about)\b',
            'objective': r'\b(objective|career objective)\b',
            'skills': r'\b(skills?|technical skills|core competencies|expertise)\b',
            'education': r'\b(education|academic|qualification)\b',
            'certifications': r'\b(certification|certificate|professional development)\b',
            'experience': r'\b(experience|employment|work history)\b',
            'projects': r'\b(projects?)\b',
            'references': r'\b(references?|referees?)\b',
            'achievements': r'\b(achievements?|awards?|honors?)\b',
            'extracurricular': r'\b(extra.?curricular|activities|leadership)\b'
        }

        for line in lines:
            line_lower = line.lower().strip()

            # Check if this line is a section header
            section_found = False
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, line_lower) and len(line_lower) < 50:
                    current_section = section_name
                    section_found = True
                    break

            # Add content to current section
            if not section_found and current_section and line.strip():
                sections[current_section] += ' ' + line

        return sections

    def get_focused_content(self, text):
        """
        Extract ONLY the content that matters for classification.
        Focus on: Summary/Objective, Skills, Education, Certifications
        IGNORE: Projects (names can mislead), References, Extracurricular
        """
        sections = self.extract_resume_sections(text)

        # Build focused content with weighted importance
        focused_parts = []

        # 1. HIGHEST PRIORITY: Summary/Objective (first paragraph context)
        if sections['summary']:
            focused_parts.append(sections['summary'] * 3)  # 3x weight
        if sections['objective']:
            focused_parts.append(sections['objective'] * 3)  # 3x weight

        # If no summary/objective, use first 500 characters as context
        if not sections['summary'] and not sections['objective']:
            first_content = ' '.join(text.split()[:100])
            focused_parts.append(first_content * 2)

        # 2. VERY HIGH PRIORITY: Skills section
        if sections['skills']:
            focused_parts.append(sections['skills'] * 4)  # 4x weight - MOST IMPORTANT

        # 3. HIGH PRIORITY: Education & Certifications
        if sections['education']:
            focused_parts.append(sections['education'] * 3)  # 3x weight
        if sections['certifications']:
            focused_parts.append(sections['certifications'] * 3)  # 3x weight

        # 4. MEDIUM PRIORITY: Experience (job titles, responsibilities)
        # Extract only job titles and core responsibilities, not project names
        if sections['experience']:
            # Remove project names that might mislead
            experience_clean = re.sub(r'project[s]?\s*:?\s*[^\n]*', '', sections['experience'], flags=re.IGNORECASE)
            focused_parts.append(experience_clean)

        # 5. IGNORE: Projects (misleading names like "MediSync"), References, Extracurricular
        # These are intentionally NOT included

        return ' '.join(focused_parts)

    def clean_text(self, text):
        """
        Clean text while preserving important context.
        Enhanced version focusing on relevant content.
        """
        if not isinstance(text, str):
            return ""

        # First, get focused content (this is the game-changer)
        focused_text = self.get_focused_content(text)

        # Convert to lowercase
        focused_text = focused_text.lower()

        # Remove URLs
        focused_text = re.sub(r'http\S+|www\S+', '', focused_text)

        # Remove email addresses (will extract separately)
        focused_text = re.sub(r'\S+@\S+', '', focused_text)

        # Remove phone numbers (will extract separately)
        focused_text = re.sub(r'[\+\(]?[0-9][0-9\s\-\(\)]{8,}[0-9]', '', focused_text)

        # Preserve important programming-related terms
        focused_text = focused_text.replace('c++', 'cplusplus')
        focused_text = focused_text.replace('c#', 'csharp')
        focused_text = focused_text.replace('.net', 'dotnet')
        focused_text = focused_text.replace('node.js', 'nodejs')
        focused_text = focused_text.replace('react.js', 'reactjs')

        # Remove special characters and digits
        focused_text = re.sub(r'[^a-zA-Z\s]', '', focused_text)

        # Remove extra whitespace
        focused_text = re.sub(r'\s+', ' ', focused_text).strip()

        return focused_text

    def preprocess_text(self, text):
        """Tokenize, remove stopwords, and lemmatize"""
        if not text or len(text.strip()) == 0:
            return ""

        try:
            # Tokenize
            tokens = word_tokenize(text)

            # Remove stopwords and lemmatize
            processed_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 2
            ]

            return ' '.join(processed_tokens)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return text

    def extract_email(self, text):
        """
        Extract email INTELLIGENTLY - avoid references section.
        Get the MAIN contact email, not referee emails.
        """
        sections = self.extract_resume_sections(text)

        # NEVER extract from references section
        text_without_references = text
        if sections['references']:
            text_without_references = text.replace(sections['references'], '')

        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

        # First, try to find email in the top portion (header area)
        # Usually the main contact email is in the first 1000 characters
        top_portion = text_without_references[:1000]
        emails_top = re.findall(email_pattern, top_portion)

        if emails_top:
            return emails_top[0]  # Return first email from top

        # If not found in top, search in full text (excluding references)
        emails_all = re.findall(email_pattern, text_without_references)

        if emails_all:
            return emails_all[0]

        return None

    # def extract_phone(self, text):
    #     """
    #     Extract phone number with INTERNATIONAL support.
    #     Handles: +94, 07X, (XXX) XXX-XXXX, etc.
    #     """
    #     sections = self.extract_resume_sections(text)
    #
    #     # NEVER extract from references section
    #     text_without_references = text
    #     if sections['references']:
    #         text_without_references = text.replace(sections['references'], '')
    #
    #     # Search in top portion first (header area)
    #     top_portion = text_without_references[:1000]
    #
    #     # Phone patterns - comprehensive
    #     phone_patterns = [
    #         # International format: +94 76 751 9740, +1 (555) 123-4567
    #         r'\+\d{1,3}\s*\(?\d{1,4}\)?\s*\d{3,4}[\s\-]?\d{3,4}',
    #         # Sri Lankan mobile: 076 751 9740, 077-1234567
    #         r'\b0[7-9]\d[\s\-]?\d{3,4}[\s\-]?\d{4}\b',
    #         # Standard formats: (555) 123-4567, 555-123-4567, 555.123.4567
    #         r'\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}',
    #         # Simple 10 digit: 1234567890
    #         r'\b\d{10}\b'
    #     ]
    #
    #     # Try each pattern on top portion first
    #     for pattern in phone_patterns:
    #         matches = re.findall(pattern, top_portion)
    #         if matches:
    #             # Clean up the match
    #             phone = matches[0]
    #             # Remove extra spaces and format
    #             phone = re.sub(r'\s+', ' ', phone).strip()
    #             return phone
    #
    #     # If not found in top, search full text (excluding references)
    #     for pattern in phone_patterns:
    #         matches = re.findall(pattern, text_without_references)
    #         if matches:
    #             phone = matches[0]
    #             phone = re.sub(r'\s+', ' ', phone).strip()
    #             return phone
    #
    #     return None
    def extract_phone(self, text):
        """
        Extract phone number prioritising:
          1. International format starting with + (e.g. +94 76 751 9740)
          2. Sri Lankan local format starting with 07 (e.g. 076 751 9740, 077-1234567)
          3. Generic international / US formats as a fallback
        Avoids extracting numbers from the references section.
        """
        sections = self.extract_resume_sections(text)

        # Never extract from references section
        text_without_references = text
        if sections['references']:
            text_without_references = text.replace(sections['references'], '')

        # Search header area first (top 1000 chars), then full text
        search_zones = [
            text_without_references[:1000],
            text_without_references
        ]

        # -----------------------------------------------------------
        # Phone patterns — ordered by PRIORITY (most specific first)
        # -----------------------------------------------------------
        phone_patterns = [
            # PRIORITY 1: International format starting with +
            # Covers: +94 76 751 9740 | +94-76-7519740 | +1 (555) 123-4567
            # Requires country code (1–3 digits) + at least 7 more digits
            r'\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}[\s\-]?\d{0,4}',

            # PRIORITY 2: Sri Lankan local — must start with 07
            # Covers: 071 234 5678 | 076-7519740 | 0771234567
            r'\b07\d[\s\-]?\d{3,4}[\s\-]?\d{4}\b',

            # PRIORITY 3: Generic 10-digit with separators (fallback)
            # Covers: (555) 123-4567 | 555-123-4567 | 555.123.4567
            # Anchored to avoid partial matches
            r'\b\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}\b',
        ]

        for zone in search_zones:
            for pattern in phone_patterns:
                matches = re.findall(pattern, zone)
                if matches:
                    phone = matches[0].strip()
                    # Normalise internal whitespace
                    phone = re.sub(r'\s+', ' ', phone)
                    # Drop trailing lone digits that got caught by the + pattern's optional group
                    phone = phone.rstrip()
                    return phone

        return None

    def extract_skills(self, text):
        """
        Extract technical skills from resume.
        Comprehensive skill detection for all job categories.
        """
        # Get the skills section specifically
        sections = self.extract_resume_sections(text)

        # Focus on skills section, but also check full text
        skills_text = (sections['skills'] + ' ' + text).lower()

        # Comprehensive skills database by category
        skills_keywords = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby',
            'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'r', 'matlab',
            'dart', 'flutter',

            # Data Science & AI
            'machine learning', 'deep learning', 'nlp', 'natural language processing',
            'data science', 'artificial intelligence', 'computer vision',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'matplotlib', 'seaborn', 'data analysis', 'data visualization',

            # Web Development
            'react', 'angular', 'vue', 'svelte', 'node.js', 'express',
            'django', 'flask', 'spring boot', 'asp.net', 'laravel',
            'html', 'css', 'bootstrap', 'tailwind', 'jquery', 'sass',

            # Mobile Development
            'android', 'ios', 'react native', 'flutter', 'xamarin',

            # Databases
            'sql', 'nosql', 'mysql', 'postgresql', 'mongodb', 'oracle',
            'redis', 'cassandra', 'dynamodb', 'firebase', 'firestore',

            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
            'jenkins', 'ci/cd', 'terraform', 'ansible', 'linux', 'bash',

            # Other Technical
            'git', 'github', 'agile', 'scrum', 'jira', 'rest api',
            'graphql', 'microservices', 'oauth', 'jwt',

            # Business & Finance
            'excel', 'powerpoint', 'word', 'sap', 'salesforce', 'crm',
            'quickbooks', 'financial modeling', 'accounting', 'bookkeeping',
            'budgeting', 'forecasting', 'financial analysis',

            # Design
            'photoshop', 'illustrator', 'figma', 'sketch', 'adobe xd',
            'indesign', 'ui/ux', 'graphic design', 'web design',

            # Healthcare (specific)
            'patient care', 'medical records', 'emr', 'clinical',
            'healthcare management', 'medical billing',

            # Engineering
            'autocad', 'solidworks', 'catia', 'ansys', 'matlab',
            'circuit design', 'embedded systems',

            # Marketing & Sales
            'digital marketing', 'seo', 'sem', 'content marketing',
            'social media marketing', 'email marketing', 'crm',

            # Soft Skills (relevant for all)
            'leadership', 'communication', 'teamwork', 'problem solving',
            'project management', 'time management', 'analytical thinking'
        }

        found_skills = []
        for skill in skills_keywords:
            if skill in skills_text:
                found_skills.append(skill)

        # Remove duplicates and return
        return list(set(found_skills))