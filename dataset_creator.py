# Import libraries
import re
import json
import random
import numpy as np
import fitz  # PyMuPDF
import faiss
import time
import torch
import pandas as pd
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import spacy
import os
import sys

# Load spaCy for better text splitting
nlp = spacy.load("en_core_web_sm")

# Define the DatasetCreator class
class DatasetCreator:
    def __init__(self):
        print("🚀 Initializing DatasetCreator: Loading NLP models... This may take a moment.")
        # Load FLAN-T5 for question generation
        self.qa_model = pipeline("text2text-generation", model="google/flan-t5-large")
        # Load T5 for paraphrasing
        self.paraphrase_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.paraphrase_model = T5ForConditionalGeneration.from_pretrained("t5-base")

        # SentenceTransformer for embeddings
        self.sentence_transformer = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        # Cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # LangChain embeddings
        self.lc_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Expanded synonym database
        self.synonyms = {
            "explain": ["describe", "elaborate on", "clarify", "detail", "expound"],
            "how": ["in what way", "by what method", "through what process", "what steps", "how does"],
            "what": ["which", "tell me about", "provide details on", "what is", "what are"],
            "why": ["for what reason", "what causes", "due to what", "what’s the purpose", "how come"],
            "describe": ["explain", "illustrate", "outline", "depict", "portray"],
            "list": ["enumerate", "name", "catalog", "itemize", "specify"],
            "compare": ["contrast", "differentiate", "juxtapose", "weigh", "match"],
            "analyze": ["examine", "scrutinize", "dissect", "evaluate", "break down"],
            "define": ["specify", "clarify", "outline meaning of", "state", "interpret"]
        }
        print("✅ Models loaded successfully!")

    def extract_text_from_file(self, file_path):
        """Extract text from various file formats."""
        with open(file_path, 'rb') as f:
            file_content = f.read()
        file_name = os.path.basename(file_path)
        if file_name.endswith(".pdf"):
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            return text.strip()
        elif file_name.endswith(".txt"):
            return file_content.decode("utf-8").strip()
        elif file_name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_content))
            return df.to_string(index=False).strip()
        else:
            raise ValueError(f"Unsupported file format: {file_name}")

    def clean_text(self, text):
        """Clean text by removing unwanted elements."""
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()

    def chunk_text_sentences(self, text):
        """Split text into chunks with spaCy."""
        if not text.strip():
            return []
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return self.text_splitter.split_text(" ".join(sentences))

    def generate_embeddings(self, text_chunks):
        """Generate embeddings for text chunks."""
        if not text_chunks:
            raise ValueError("No text chunks provided.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentence_transformer.to(device)
        embeddings = self.sentence_transformer.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

    def create_faiss_index(self, embeddings):
        """Create FAISS index for similarity search."""
        if embeddings.size == 0:
            raise ValueError("Embeddings array is empty.")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def create_langchain_vectorstore(self, text_chunks):
        """Create LangChain FAISS vectorstore."""
        return LangchainFAISS.from_texts(texts=text_chunks, embedding=self.lc_embeddings)

    def retrieve_relevant_context(self, query, index, embeddings, text_chunks, top_k=5):
        """Retrieve relevant context using FAISS and re-ranking."""
        query_embedding = self.sentence_transformer.encode([query])
        distances, indices = index.search(query_embedding, top_k)
        relevant_chunks = [text_chunks[idx] for idx in indices[0]]
        scores = self.cross_encoder.predict([(query, chunk) for chunk in relevant_chunks])
        return [chunk for _, chunk in sorted(zip(scores, relevant_chunks), reverse=True)][:3]

    def generate_questions(self, text, num_questions=5):
        """Generate diverse questions using FLAN-T5."""
        input_text = f"Generate {num_questions} diverse, specific questions based on: {text}"
        result = self.qa_model(input_text, max_length=60, num_beams=num_questions+2, num_return_sequences=num_questions)
        return [r["generated_text"].strip() for r in result]

    def paraphrase_question(self, question):
        """Generate a paraphrase of the question using T5."""
        input_text = f"paraphrase: {question}"
        input_ids = self.paraphrase_tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.paraphrase_model.generate(input_ids, max_length=60, num_beams=5, early_stopping=True)
        return self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def augment_question(self, question, context, augmentation_factor=3):
        """Augment questions with synonyms and paraphrasing."""
        augmented_questions = {question}  # Use set to avoid duplicates

        # Synonym replacement
        words = question.split()
        replaceable = [(i, w) for i, w in enumerate(words) if w.lower() in self.synonyms]
        for _ in range(augmentation_factor):
            if replaceable:
                new_words = words.copy()
                pos, word = random.choice(replaceable)
                context_embedding = self.sentence_transformer.encode([context])
                synonym_embeddings = self.sentence_transformer.encode(self.synonyms[word.lower()])
                similarities = cosine_similarity(context_embedding, synonym_embeddings)
                best_synonym = self.synonyms[word.lower()][np.argmax(similarities)]
                new_words[pos] = best_synonym
                augmented_questions.add(" ".join(new_words))

        # Add paraphrase
        paraphrased = self.paraphrase_question(question)
        augmented_questions.add(paraphrased)

        return list(augmented_questions)

    def filter_similar_questions(self, questions, threshold=0.75):
        """Filter out similar questions."""
        if not questions:
            return []
        embeddings = self.sentence_transformer.encode(questions)
        similarity_matrix = cosine_similarity(embeddings)
        filtered = []
        for i, q in enumerate(questions):
            if all(similarity_matrix[i][j] < threshold for j in range(i)):
                filtered.append(q)
        return filtered

    def create_qa_dataset(self, text_chunks, embeddings, faiss_index, use_rag=True, in_context_learning=True):
        """Create QA dataset with augmentation."""
        dataset = []
        vectorstore = self.create_langchain_vectorstore(text_chunks) if in_context_learning else None
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) if in_context_learning else None
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if in_context_learning else None

        for chunk in text_chunks:
            questions = self.generate_questions(chunk, num_questions=5)
            questions = self.filter_similar_questions(questions)

            for q in questions:
                augmented_qs = self.augment_question(q, chunk)
                for aug_q in augmented_qs:
                    if use_rag:
                        context = " ".join(self.retrieve_relevant_context(aug_q, faiss_index, embeddings, text_chunks)) if not in_context_learning else " ".join([doc.page_content for doc in retriever.get_relevant_documents(aug_q)])
                    else:
                        context = chunk
                    context = self.post_process(context)
                    dataset.append({"role": "user", "content": aug_q})
                    dataset.append({"role": "assistant", "content": context})

        return dataset

    def post_process(self, text):
        """Enhance text quality."""
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[0].upper() + text[1:] if text else text
        if not text.endswith(('.', '!', '?')):
            text += '.'
        return text

    def process_files(self, file_paths, chunk_size=1500, use_rag=True, similarity_threshold=0.75, in_context_learning=True):
        """Process multiple file types and generate dataset."""
        print("📂 Step 1/5: Processing files...")
        extracted_text = ""
        total_files = len(file_paths)
        for i, file_path in enumerate(file_paths, 1):
            print(f"⏳ Processing file {i}/{total_files}: {file_path}")
            try:
                text = self.extract_text_from_file(file_path)
                if not text:
                    print(f"⚠️ No text extracted from {file_path}. Skipping.")
                    continue
                extracted_text += text + "\n\n"
                print(f"✔️ Successfully extracted text from {file_path}")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")

        if not extracted_text.strip():
            print("❌ No valid text extracted from any files. Aborting.")
            return None, None, None

        print("🧹 Step 2/5: Cleaning extracted text...")
        extracted_text = self.clean_text(extracted_text)
        print(f"📏 Extracted text length: {len(extracted_text)} characters")

        print("✂️ Step 3/5: Chunking text into manageable pieces...")
        text_chunks = self.chunk_text_sentences(extracted_text)
        if not text_chunks:
            print("❌ No text chunks generated. Check input content.")
            return None, None, None
        print(f"✅ Generated {len(text_chunks)} text chunks")

        print("🔢 Step 4/5: Generating embeddings...")
        embeddings = self.generate_embeddings(text_chunks)
        print(f"✅ Embeddings generated with shape: {embeddings.shape}")

        print("📈 Step 5/5: Creating FAISS index and QA dataset...")
        faiss_index = self.create_faiss_index(embeddings)
        dataset = self.create_qa_dataset(text_chunks, embeddings, faiss_index, use_rag, in_context_learning)
        print("🎉 Dataset creation completed!")
        return extracted_text, text_chunks, dataset

# Main function for local execution
def main():
    print("📋 Dataset Creator for Local Environment")
    print("ℹ️ Provide file paths as command-line arguments (e.g., python dataset_creator.py file1.pdf file2.txt)")
    
    if len(sys.argv) < 2:
        print("❌ Error: No files provided. Please specify at least one file path.")
        print("Example: python dataset_creator.py sample.pdf sample.txt")
        sys.exit(1)

    file_paths = sys.argv[1:]
    invalid_files = [f for f in file_paths if not os.path.exists(f)]
    if invalid_files:
        print(f"❌ Error: The following files do not exist: {invalid_files}")
        sys.exit(1)

    # Settings (hardcoded for simplicity; you can modify these)
    chunk_size = 1500
    use_rag = True
    in_context_learning = True
    similarity_threshold = 0.75

    print("🚀 Starting processing pipeline...")
    start_time = time.time()

    # Initialize DatasetCreator
    creator = DatasetCreator()

    # Process files
    extracted_text, text_chunks, dataset = creator.process_files(
        file_paths,
        chunk_size=chunk_size,
        use_rag=use_rag,
        similarity_threshold=similarity_threshold,
        in_context_learning=in_context_learning
    )

    if dataset is None:
        print("❌ Processing failed. Check the logs above for details.")
        sys.exit(1)

    # Save dataset
    output_file = "dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"✅ Dataset saved to '{output_file}'")

    # Calculate processing time
    processing_time = time.time() - start_time

    # Display results
    print("\n" + "="*50)
    print(f"🎉 Processing completed in {processing_time:.2f} seconds!")
    print(f"📊 Number of text chunks: {len(text_chunks)}")
    print(f"❓ Number of QA pairs: {len(dataset) // 2}")
    print("="*50 + "\n")

    # Display sample QA pairs
    sample_size = min(5, len(dataset) // 2)
    if sample_size > 0:
        print("📋 Sample Q&A Pairs:")
        for i in range(sample_size):
            question = dataset[i * 2]["content"]
            answer = dataset[i * 2 + 1]["content"]
            truncated_answer = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"Q: {question}")
            print(f"A: {truncated_answer}\n")
    else:
        print("⚠️ No QA pairs generated. Dataset may be empty.")

if __name__ == "__main__":
    main()