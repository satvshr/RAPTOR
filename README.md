# VelociRAPTOR🦖: A RAPTOR RAG Implementation from Scratch

Welcome to **VelociRAPTOR**—an end-to-end project combining **RAPTOR RAG** (Retrieval Augmented Generation) techniques with a custom suite of algorithms like GMM, UMAP, EM, and the BIC criterion. Everything here is implemented from scratch. Using **NumPy**.

We are using the **llama-3.2-3b-instruct** model from **LMStudio** as the LLM Agent, and **bart-large-cnn** from Facebook as the summarizer.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Setup and Installation](#setup-and-installation)  
3. [Environment Variables](#environment-variables)  
4. [Why Two Directories?](#why-two-directories)  
5. [File Descriptions](#file-descriptions)  
6. [Demo Video](#demo-video)

---

## Project Overview

**VelociRAPTOR** brings together the power of:

- **RAPTOR RAG** for retrieval-augmented generation  
- **Gaussian Mixture Models (GMM)** for clustering  
- **Uniform Manifold Approximation and Projection (UMAP)** for dimensionality reduction  
- **Expectation-Maximization (EM)** for iterative parameter estimation (used to implement GMM at its core)  
- **Bayesian Information Criterion (BIC)** to evaluate model complexity (used in GMM to get optimal number of clusters to be used by the algorithm)

All implemented in pure **NumPy**!

In this project, we employ **llama-3.2-3b-instruct** (via **LMStudio**) for LLM-based text generation and **bart-large-cnn** (by Facebook) for summarization tasks.

---

## Setup and Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/satvshr/VelociRAPTOR.git
   cd VelociRAPTOR
   ```

2. **Create a Virtual Environment** (optional but recommended)  
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Linux/Mac
   # or
   venv\Scripts\activate      # On Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip install numpy langchain pydf2 ast re os langchain_community pydantic python-dotenv requests
   ```

4. **Configure Environment**  
   - Create a file named `.env` in the project root. [See the **Environment Variables**](#environment-variables) for details on what to include.

---

## Environment Variables

Create a file called `.env` in the project root (same level as `README.md`). It should contain:

```
LANGSMITH_API_KEY= <Your Langsmith API Key>
SUMMARIZER_API_KEY= <Your Hugging Face API Key>
LANGCHAIN_TRACING_V2=true
FILE_PATH= <Parent path of the files you want to upload>
```

- **LANGSMITH_API_KEY**: API key used for LangSmith integration (if applicable).  
- **SUMMARIZER_API_KEY**: Your Hugging Face API Key.  
- **LANGCHAIN_TRACING_V2**: A boolean-like flag (true/false) to enable or disable advanced tracing in LangChain.  
- **FILE_PATH**: The path to the files you want to upload or process.

---

## Why Two Directories?

- **`src`**: Contains the core scripts of the application. These files form the main workflow and logic for generation, retrieval, and advanced RAG features.  
- **`utils`**: Provides auxiliary helper scripts—smaller modules or specialized utilities like GMM, UMAP, or PDF summarization—enhancing your core pipeline without cluttering it.

Splitting into `src` and `utils` keeps the architecture organized and allows for focused development in each area.

---

## File Descriptions

### `src` Directory

- **generation.py**  
  Handles generation using the question asked and the output received after RAG by prompting the LLM Agent.

- **indexing.py**  
  Responsible for indexing documents. This involves creating vector embeddings and storing them in a Chroma vectorstore.

- **main.py**  
  The primary entry point for the project. Run this file to launch the main application or orchestrate different modules.

- **raptor.py**  
  Implements RAPTOR logic, leveraging retrieval-augmented generation and the custom algorithms implemented from scratch.

- **retrieval.py**  
  Handles retrieving documents or information from the created index.

- **routing.py**  
  Chooses the most relevant files based on the cosine similarity with the question.

- **translation.py**  
  Implements multi-query generation by using the LLM Agent.

### `utils` Directory

- **find_documents.py**  
  Searches or scans a directory structure to locate relevant documents.

- **gmm.py**  
  Contains a pure NumPy implementation of **Gaussian Mixture Models**, including **EM** and **BIC** for determining the optimal number of components.

- **lm_studio.py**  
  A place for experimenting with language models (LMs) or hosting a local interface for LM experimentation.

- **pdf_summarizer.py**  
  Summarizes PDFs by extracting text, chunking it, and optionally using **bart-large-cnn** from Facebook to produce concise summaries.

- **umap.py**  
  Implements **UMAP** from scratch in NumPy for dimensionality reduction, simplified to make it easier to implement.

---

## Demo Video
<video src='https://github.com/user-attachments/assets/549abeea-9be2-4720-b945-914cc7248a2d' width=180/>

---
