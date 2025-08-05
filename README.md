# Twitter Sentiment Analysis Project

This project implements sentiment analysis on Twitter data using machine learning techniques. The analysis classifies tweets into positive (4) and negative (0) sentiments using Support Vector Machine (SVM) classification.

## Project Overview

This project analyzes Twitter sentiment by processing raw tweets through various natural language processing techniques and applying machine learning to classify the emotional tone of the tweets.

## Project Structure

The project is organized into the following main sections:

1. Data Pre-Processing
2. Model Training
3. Results Analysis

## Detailed Implementation

### 1. Data Pre-Processing

#### Libraries Used
- pandas: For data manipulation and DataFrame operations
- matplotlib: For data visualization and plotting
- nltk: For natural language processing tasks
- re: For regular expression operations
- collections: For word frequency counting

#### Steps:
1. Data Loading
   - Reads CSV file containing Twitter data
   - Samples 1% of data (approximately 16,000 entries)
   - Selects only Target (sentiment) and Tweet columns
   - Balances dataset between positive and negative classes

2. Text Preprocessing
   - Removes special characters, numbers, and punctuation using regex
   - Tokenizes text into individual words using NLTK
   - Converts all text to lowercase for consistency
   - Removes common stopwords (e.g., "the", "is", "at")
   - Applies lemmatization to reduce words to their base form
   - Generates n-grams (specifically bigrams) for feature engineering
   - Implements custom normalizer function for text cleaning

### 2. Model Training

#### Libraries Used
- scikit-learn: For machine learning implementations
- scipy: For sparse matrix operations
- numpy: For numerical operations

#### Steps:
1. Data Vectorization
   - Uses CountVectorizer for text-to-numeric conversion
   - Parameters:
     - ngram_range=(1,2) for unigrams and bigrams
   - Creates sparse matrix representation
   - Indexes vectorized data for tracking

2. Train-Test Split
   - Splits data into 60% training and 40% testing sets
   - Preserves random state for reproducibility
   - Maintains class distribution in split

3. Model Implementation
   - Uses OneVsRestClassifier with SVM
   - Model Parameters:
     - gamma: 0.01 (kernel coefficient)
     - C: 100 (regularization parameter)
     - kernel: linear
     - class_weight: balanced (handles class imbalance)
     - probability: True (enables probability estimates)

### 3. Results Analysis

#### Evaluation Metrics
1. Confusion Matrix Results:
   ```
   True Negatives:  2,584 (32.26%)
   False Positives: 654   (8.17%)
   False Negatives: 713   (8.90%)
   True Positives:  4,061 (50.67%)
   ```

2. Classification Report:
   ```
   Classification Metrics:
                 Precision    Recall  F1-Score   Support
   Negative       0.78        0.86    0.82      2,998
   Positive       0.86        0.78    0.82      3,014
   
   Accuracy                          0.82       6,012
   Macro avg      0.82        0.82   0.82      6,012
   Weighted avg   0.82        0.82   0.82      6,012
   ```

3. Key Performance Indicators:
   - Overall Model Accuracy: 82%
   - Balanced performance across both positive and negative classes
   - Strong F1-Score indicating good balance between precision and recall

## Running the Project

1. Install Required Dependencies:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn
   ```

2. NLTK Downloads:
   ```python
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

3. Environment Setup:
   - Mount Google Drive (if using Google Colab)
   - Update dataset path to your local path
   - Ensure Python 3.x environment

4. Execution:
   - Run notebook cells in sequential order
   - Monitor output at each step for verification

## Technical Notes

- Model Configuration:
  - Uses balanced class weights to handle any class imbalance
  - Linear kernel SVM for efficient high-dimensional data processing
  - Probability estimates enabled for detailed analysis

- Data Processing:
  - Extensive text cleaning pipeline
  - N-gram feature engineering for context capture
  - Sparse matrix operations for memory efficiency

- Performance Considerations:
  - Sampling used for large datasets
  - Vectorized operations for efficiency
  - Optimized SVM parameters for sentiment classification
