# Semantic NLP Filtering and Classification


## Overview
This project implements a semantic NLP approach to filter, classify, and extract information from a collection of academic papers. The goal is to identify papers that apply deep learning techniques in virology and epidemiology.


### Semantic Filtering
  - Use TF-IDF vectorization to convert the text of abstracts into numerical vectors.
  - Cosine similarity is calculated between each abstract and deep learning keywords.
  - Papers with similarity score above specified threshold are considered.

### Classification of Papers
  - Use keywords for "text mining" and "computer vision" as give [Here](https://docs.google.com/document/d/1uMkXik3B3rNnKLbZc5AyqWruTGUKdpJcZFZZ4euM0Aw/edit?tab=t.0#heading=h.gjdgxs).
  - Papers are classified based on these keywords present in abstracts.

### Task 3: Method Extraction
  - Check for mentions of common deep learning models. If methods found its been recorded in separate column.

## Running the Code
Make sure to have the necessary libraries installed:
- `pandas`
- `sklearn`

## Output
- Output is saved in `classified_and_methods_with_runtime.csv`, which includes:
  - **Title**: Title of paper.
  - **Category**: Classification category ("text mining," "computer vision," "both," or "other").
  - **Methods**: Names of deep learning methods found.
  - **Abstract**: Original abstract just for ref.

## Working
1. Data is loaded and cleaned to prepare for analysis.
2. TF-IDF and cosine similarity used to filter relevant papers.
3. Papers categorized based on keyword.
4. Deep learning methods are extracted from abstracts.
5. Script measures and prints running time.