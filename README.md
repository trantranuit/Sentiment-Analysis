# Multilingual Sentiment Analysis

## Overview
This project implements sentiment analysis on multilingual text data (English and Spanish) using various machine learning approaches. The analysis is performed for both binary classification (positive/negative) and multi-class sentiment classification. The project investigates the effectiveness of different text representation techniques with multiple classification algorithms to determine optimal approaches for sentiment analysis in multiple languages.

## Dataset
The dataset contains text samples in both English and Spanish with sentiment annotations. Each sample is labeled with:
- Binary sentiment (positive/negative)
- Multi-class sentiment (specific emotion categories)

Data files are structured as follows:
- Training data: `data/en_train.csv`, `data/es_train.csv`
- Development data: `data/en_dev.csv`, `data/es_dev.csv`

### Data characteristics:
- Text samples of varying lengths
- Binary class distribution: The data includes both positive and negative sentiment labels
- Multi-class distribution: Contains multiple sentiment categories
- Two languages: English and Spanish

## Project Structure



## Features

### Exploratory Data Analysis (EDA)
The project begins with a comprehensive exploratory data analysis:
- Basic data statistics and information
- Visualization of label distribution for binary and multi-class tasks
- Text length analysis with histograms showing distribution
- Word frequency analysis identifying common terms
- Word clouds for visual representation of corpus content
- Missing values analysis
- Comparison between English and Spanish datasets

### Preprocessing Steps
Text preprocessing pipeline includes:
- URL removal via regular expressions
- Special character and punctuation removal
- Case normalization (converting to lowercase)
- Tokenization using NLTK's word_tokenize
- Stopword removal (language-specific) using NLTK's stopwords corpus
- Label encoding for classification targets

### Feature Engineering
Four different text vectorization methods are implemented and compared:

1. **Bag of Words (BOW)** 
   - Implemented using CountVectorizer
   - Parameters: max_features=10000, ngram_range=(1,3)
   - Simple frequency-based representation

2. **TF-IDF** (Term Frequency-Inverse Document Frequency)
   - Implemented using TfidfVectorizer
   - Parameters: max_features=10000, ngram_range=(1,3)
   - Weights terms by their importance in the corpus

3. **Word2Vec**
   - Dense word embeddings trained on the corpus
   - Parameters: vector_size=300, window=7, min_count=3
   - Skip-gram architecture (default)
   - Document vectors created by averaging word vectors

4. **Doc2Vec** (for binary classification only)
   - Document-level embeddings trained directly
   - Both DM (Distributed Memory) and DBOW (Distributed Bag of Words) models
   - Parameters: vector_size=300, window=10, min_count=4, epochs=50
   - Combined representation using weighted average of both models

## Machine Learning Models
Four different classification algorithms are implemented and compared:

1. **Support Vector Machines (SVM)**:
   - Linear kernel SVM for all vectorization methods
   - RBF kernel SVM with hyperparameter tuning (C and gamma)
   - GridSearchCV for optimal parameter selection

2. **Naive Bayes**:
   - MultinomialNB for BOW and TF-IDF features
   - GaussianNB for Word2Vec and Doc2Vec features
   - Alpha parameter tuning via GridSearchCV
   - StandardScaler for non-count-based features

3. **K-Nearest Neighbors (KNN)**:
   - Using cosine similarity metric
   - Fixed k=5 and optimized k via GridSearchCV
   - Cross-validation evaluation

4. **Decision Tree**:
   - Gini impurity criterion
   - Max depth optimization via GridSearchCV
   - Comprehensive performance evaluation

## Results
The project provides comprehensive evaluation metrics for all combinations of:
- Feature engineering methods
- Classification algorithms
- Languages (English and Spanish)
- Classification tasks (Binary and Multi-class)

Performance metrics include:
- Classification reports (precision, recall, F1-score)
- Accuracy scores
- Cross-validation results
- Optimal hyperparameters for each model

### Key Findings:
- Different vectorization methods exhibit varying performance across languages
- SVM with TF-IDF generally provides robust results for both languages
- Word embeddings (Word2Vec and Doc2Vec) show strong performance with appropriate classifiers
- Language-specific variations highlight the importance of tailored approaches

## Setup and Installation
1. Clone this repository
2. Install required packages:
```python
   pip install nltk pandas matplotlib seaborn wordcloud scikit-learn gensim numpy spacy
3. Download NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
