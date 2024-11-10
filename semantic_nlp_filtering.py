import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Start timer
start_time = time.time()
file_path = 'collection_with_abstracts.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# Data Preprocessing - cleaning text
def clean_text(text):
    if pd.isnull(text):
        return ""
    # Remove special characters/digits, convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    return text

data['Cleaned_Title'] = data['Title'].apply(clean_text)
data['Cleaned_Abstract'] = data['Abstract'].apply(clean_text)

# keywords for filtering
keywords = [
    "neural network", "artificial neural network", "machine learning model", "feedforward neural network", "neural net algorithm", "multilayer perceptron",
    "convolutional neural network", "recurrent neural network", "long short-term memory network", "CNN", "GRNN", "RNN", "LSTM", "deep learning", "deep neural networks",
    "computer vision", "vision model", "image processing", "vision algorithms", "object recognition", "scene understanding", "natural language processing",
    "text mining", "NLP", "computational linguistics", "language processing", "text analysis", "generative artificial intelligence", "generative AI",
    "transformer models", "self-attention models", "transformer architecture", "attention-based neural networks", "sequence-to-sequence models","large language model",
    "llm", "transformer-based model", "pretrained language model", "generative language model", "foundation model", "state-of-the-art language model", "multimodal model",
    "multimodal neural network", "vision transformer", "diffusion model", "generative diffusion model", "diffusion-based generative model","continuous diffusion model"
]
keywords_text = " ".join(keywords)


# Semantic Filtering Using NLP, vectorize abstracts, keywords using TF-IDF
vectorizer = TfidfVectorizer()
abstract_vectors = vectorizer.fit_transform(data['Cleaned_Abstract'])
keywords_vector = vectorizer.transform([keywords_text])
cosine_similarities = cosine_similarity(abstract_vectors, keywords_vector).flatten()

# Filter papers with similarity score above threshold
threshold = 0.1

relevant_indices = []
# Iterate through each index
for i, score in enumerate(cosine_similarities):
    # Appedn if score > threshold
    if score > threshold:
        relevant_indices.append(i)

relevant_papers = data.iloc[relevant_indices].copy()

# Classification based on type of method used
def classify_paper(abstract):
    text_mining_keywords = ["natural language processing", "text mining", "NLP","computational linguistics", "language processing", "text analytics", "textual data analysis", "text data analysis", "text analysis", "speech and language technology", "language modeling", "computational semantics"]
    computer_vision_keywords = ["computer vision", "vision model", "image processing", "vision algorithms", "computer graphics and vision", "object recognition", "scene understanding"]
    
    abstract_lower = abstract.lower()
    # Initialize flags for text mining and computer vision
    is_text_mining = False
    is_computer_vision = False

    # Check text mining keywords in abstract
    for keyword in text_mining_keywords:
        if keyword in abstract_lower:
            is_text_mining = True
            break 

    # Check computer vision keywords in abstract
    for keyword in computer_vision_keywords:
        if keyword in abstract_lower:
            is_computer_vision = True
            break
    
    if is_text_mining and is_computer_vision:
        return "both"
    elif is_text_mining:
        return "text mining"
    elif is_computer_vision:
        return "computer vision"
    else:
        return "other"

# Apply classification to relevant papers
relevant_papers.loc[:, 'Category'] = relevant_papers['Cleaned_Abstract'].apply(classify_paper)

# Step 5: Extracting Deep Learning Methods
def extract_methods(abstract):
    deep_learning_methods = ["neural network", "artificial neural network", "machine learning model", "feedforward neural network", "neural net algorithm", "multilayer perceptron", "convolutional neural network", "recurrent neural network", "long short-term memory network", "CNN", "GRNN", "RNN", "LSTM"]
    found_methods = []

    # Iterate to list of deep learning methods
    for method in deep_learning_methods:
        # Check method is present in abstract
        if method in abstract:
            found_methods.append(method)
    return ", ".join(found_methods) if found_methods else "None"

# Apply method extraction function
relevant_papers.loc[:, 'Methods'] = relevant_papers['Cleaned_Abstract'].apply(extract_methods)

end_time = time.time()
# calculate running time
running_time = end_time - start_time

output_file_path = 'output_relevant_papers.csv'
relevant_papers[['Title', 'Category', 'Methods', 'Abstract']].to_csv(output_file_path, index=False)

print(f"Results written to File: {output_file_path}")
print(f"Running Time: {running_time} seconds")