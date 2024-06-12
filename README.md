

# Twitter Sentiment Analysis

This project is designed to analyze and classify the sentiment of tweets using machine learning techniques. The dataset includes tweets labeled as positive or negative, which are used to train and evaluate the models.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

This project aims to build a sentiment analysis model for Twitter data. The process includes data preprocessing, exploratory data analysis, model training, and evaluation using various machine learning algorithms.

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/roqaiaAshraf/Twitter-Sentiments
```

## Usage

1. **Clone the repository:**

```bash
git clone https://github.com/roqaiaAshraf/Twitter-Sentiments
cd twitter-sentiment-analysis
```

2. **Download the dataset:**

Place your dataset file (`Twitter Sentiments.csv`) in the `data/` directory.

3. **Run the analysis:**

You can run the analysis using Jupyter Notebook or a Python script.

```bash
jupyter notebook sentiment_analysis.ipynb
```

or

```bash
python sentiment_analysis.py
```

## Data

The dataset contains the following columns:
- `id`: Unique identifier for each tweet
- `label`: Sentiment label (0 for negative, 1 for positive)
- `tweet`: The tweet text

## Data Preprocessing

1. **Remove Patterns:**

   The function `remove_patterns` removes specific patterns from tweets, such as user mentions.

    ```python
    def remove_patterns(txt, pattern):
        words_found = re.findall(pattern, txt)
        for word in words_found:
            txt = re.sub(word, '', txt)
        return txt
    ```

2. **Cleaning Tweets:**

   - Removed user mentions.
   - Removed special characters and numbers.
   - Removed words with length less than 3.
   - Tokenized and stemmed words.

    ```python
    df['clean_tweet'] = np.vectorize(remove_patterns)(df['tweet'], "@[\w]*")
    df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ", regex=True)
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))
    ```

3. **Tokenization and Stemming:**

    ```python
    tokenized_tweets = df['clean_tweet'].apply(lambda x: x.split())
    stemmer = PorterStemmer()
    tokenized_tweets = tokenized_tweets.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
    for i in range(len(tokenized_tweets)):
        tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
    df['clean_tweet'] = tokenized_tweets
    ```

4. **Visualize Words:**

   Generate word clouds to visualize the most frequent words in positive and negative tweets.

    ```python
    all_words = ' '.join([sentence for sentence in df['clean_tweet']])
    wordcloud = WordCloud(height=400, width=600, random_state=42, max_font_size=100).generate(all_words)
    plt.figure(figsize=(15,18))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    ```

## Model Training

### Machine Learning Models Used:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Multinomial Naive Bayes**

### Workflow:

1. **Vectorization:**

    ```python
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)
    ```

2. **Handling Imbalanced Data:**

    ```python
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X_tfidf, y)
    ```

3. **Train-Test Split:**

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    ```

4. **Model Training and Evaluation:**

    ```python
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Classification Report:\n", classification_report(y_test, pred))
    ```

## Results

### Logistic Regression
- **Accuracy:** 0.95
- **Classification Report:**
  ```
                precision    recall  f1-score   support

            0       0.98      0.93      0.95      5892
            1       0.93      0.98      0.96      5996

     accuracy                           0.95     11888
    macro avg       0.96      0.95      0.95     11888
 weighted avg       0.96      0.95      0.95     11888
  ```

### Random Forest
- **Accuracy:** 0.99
- **Classification Report:**
  ```
                precision    recall  f1-score   support

            0       0.99      0.98      0.99      5892
            1       0.98      0.99      0.99      5996

     accuracy                           0.99     11888
    macro avg       0.99      0.99      0.99     11888
 weighted avg       0.99      0.99      0.99     11888
  ```

### Multinomial Naive Bayes
- **Accuracy:** 0.94
- **Classification Report:**
  ```
                precision    recall  f1-score   support

            0       0.97      0.91      0.94      5892
            1       0.92      0.98      0.95      5996

     accuracy                           0.94     11888
    macro avg       0.95      0.94      0.94     11888
 weighted avg       0.95      0.94      0.94     11888
  ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you have any suggestions or improvements.
