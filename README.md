# Learn AI and Data Science

![image](https://github.com/user-attachments/assets/3012ba68-fe35-47b9-b974-015c50d72372)


## Introduction
This project is titled **Learn AI and Data Science** and focuses on building an information retrieval system to assist users in finding relevant content from a web domain. The system is developed using the following steps:

1. **Web Crawling**: Collecting data from GeeksforGeeks.
2. **Text Preprocessing and Indexing**: Cleaning and organizing text to create an inverted index.
3. **Information Retrieval**: Implementing the Vector Space Model (VSM) and Query Likelihood Model (QLM) to retrieve and rank documents based on user queries.

---

## Features
- **Web Crawling**: Scrapes 250+ pages from GeeksforGeeks, focusing on AI and Data Science.
- **Text Preprocessing**: Utilizes tokenization, lemmatization, and normalization to clean data.
- **Inverted Index**: Maps terms to document IDs and their positions for fast retrieval.
- **Information Retrieval Models**:
  - **Vector Space Model (VSM)**: Uses TF-IDF and cosine similarity to rank documents.
  - **Query Likelihood Model (QLM)**: Uses language models and smoothing techniques for probabilistic ranking.
- **User Interface**: A dash based application with tabs for VSM and QLM results.

---

## Project Workflow

### 1. Web Crawling
- **Objective**: Collect 250+ web pages starting from a seed URL.
- **Script**: `crawl.py`
- **Features**:
  - Starts crawling from the given seed page.
  - Collects content from headers and paragraphs.
  - Saves each page as a text file in the `Documents` directory.
  - Implements error handling and delays to prevent server overload.

### 2. Text Preprocessing and Inverted Index
- **Objective**: Prepare text for indexing and create a searchable structure.
- **Script**: `invertedindex.py`
- **Steps**:
  - Tokenization: Splits text into individual words.
  - Lemmatization: Converts words to their base forms.
  - Inverted Index: Maps each word to the documents it appears in and their positions.
  - Saves the index as `inverted_index.json`.

### 3. Information Retrieval
- **Objective**: Implement search models to rank documents based on relevance to user queries.
- **Script**: `app.py`
- **Models**:
  1. **Vector Space Model (VSM)**:
     - Computes TF-IDF for terms.
     - Represents documents and queries as vectors.
     - Calculates cosine similarity to rank documents.
  2. **Query Likelihood Model (QLM)**:
     - Builds a language model for each document.
     - Applies smoothing techniques (Linear Interpolation).
     - Calculates query likelihood scores to rank documents.

---


