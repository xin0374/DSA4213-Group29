# DSA4213-Group9

## 📊 Dataset

This project uses the [MedQuAD Medical Question Answer Dataset](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research) from Kaggle. 

It is a curated collection of over **16,000 medical question-answer** pairs meticulously curated from 12 trusted National Institutes of Health (NIH) websites. It is designed for medical question answering, NLP research, and healthcare chatbot applications. 

This project focuses on **building a chatbot for medical queries** using **Large Language Models (LLMs)** combined with **Retrieval-Augmented Generation (RAG)** techniques.  The chatbot is designed to retrieve accurate, fact-based medical information from the dataset and generate human-like, context-aware responses for users seeking healthcare knowledge.

## Setup instructions

### 1. Clone repo

```bash
git clone git@github.com:xin0374/DSA4213-Group29.git

cd DSA4213-Group29
```

### 2. Install dependencies 

Ensure you have Python 3.9+ installed, then run:

```
pip install -r requirements.txt
```

### 3. Training pipeline 

Step 1: Precompute contexts for the dataset

```
python precompute_contexts.py
```

Step 2: Fine-tune BioGPT with LoRA on the precomputed contexts

Train the model on precomputed contexts

```
python finetune_biogpt.py
```

### 4. Evaluating on validation dataset 

Assess models performance using validation split 
- Baseline 1: Fine-tuned BioGPT
- Baseline 2: Retrieval + Zero-shot BioGPT
- Full variant: Retrieval + Fine-tuned BioGPT

```
python run_evaluation.py
```

### 5. Testing the model via chatbot interface (Streamlit)

1. Download [base BioGPT](https://drive.google.com/drive/folders/1ho36q9dWZsLYQ9T9iXbsDO3-soS8Ebgj?usp=drive_link), [fine-tuned BioGPT](https://drive.google.com/drive/folders/1CS1vFI0IoVe4SYivi4LlzkvoRQ3MLUn4?usp=drive_link) and [retrieval artifacts](https://drive.google.com/drive/folders/1tvgXqAdbUKWgi0IWqgZ2BQdhE7sTdW8r?usp=drive_link) from google drive and save them inside this repo folder in your local directory

2. Run the Streamlit chabot:

```
streamlit run chatbot.py
```
