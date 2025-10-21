# 🐢 TURTLE AUGMENTOR
### Dataset Augmentation for Fine-Tuning LLMs using RAG

## Overview
**Turtle Augmentor** is an AI-powered tool designed to automate the conversion of unstructured textual content (like PDFs and research papers) into **structured conversational datasets** in a Question-Answer (Q&A) format.

These high-quality datasets are optimized for **fine-tuning Large Language Models (LLMs)** and training AI chatbots. The system uses **Retrieval Augmented Generation (RAG)** to enhance contextual coherence and accuracy in the generated data.

---

## Key Technologies
The project uses a modular architecture built on advanced NLP and retrieval tools:
* **Augmentation/Generation:** Uses the **Flan-T5** model for automated question generation.
* **Retrieval:** Integrates **FAISS** for efficient similarity search and **LangChain** for enhanced contextual retrieval.
* **Processing:** Employs **PyMuPDF** for document parsing and recursive text splitting.
* **Architecture:** Uses a **Python Backend API** with a Vite/React.js Frontend UI.

---

## Setup and Running

To set up and run the application, execute the following commands in your terminal:

1.  **Initialize the Git repository:**
    ```bash
    git init
    ```
2.  **Clone the project repository:**
    ```bash
    git clone https://github.com/ABIRAM323/TURTLE-AUGMENTOR.git
    ```
3.  **Navigate into the project directory:**
    ```bash
    cd TURTLE-AUGMENTOR
    ```
4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```    
5.  **Run the main application file:**
    ```bash
    run python app.py
    ```

---

## Important Note

**Better run it in VS Code (note!)**