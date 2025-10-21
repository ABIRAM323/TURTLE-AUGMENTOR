# dataset_creator_fixed.py
# Corrected & hardened version of user's script

# Standard libs
import io
import re
import json
import random
import os
import sys
import time
from typing import List, Tuple, Optional

# Third-party libs
import numpy as np
import pandas as pd
import torch

# PyMuPDF module is imported as `fitz`
import fitz

# faiss (make sure the right faiss package is installed for your platform)
import faiss

from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# LangChain (community) - keep as-is but may need version compatibility
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import spacy

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model missing, give a clear error
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")

class DatasetCreator:
    def __init__(self):
        print("🚀 Initializing DatasetCreator: Loading NLP models... This may take a moment.")
        # --- models / pipelines ---
        # FLAN-T5 pipeline for question generation (text2text)
        try:
            self.qa_model = pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            # fallback / clearer error
            raise RuntimeError(f"Failed to load FLAN-T5 pipeline: {e}")

        # Paraphrase T5
        self.paraphrase_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.paraphrase_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        # Put T5 on appropriate device
        if torch.cuda.is_available():
            self.paraphrase_model.to("cuda")

        # Sentence transformer for embeddings
        self.sentence_transformer = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        if torch.cuda.is_available():
            try:
                self.sentence_transformer = self.sentence_transformer.to('cuda')
            except Exception:
                # some versions require .to on underlying model; ignore if not available
                pass

        # Cross-encoder for reranking
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

        # LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # LangChain embeddings wrapper
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

    # ---------- I/O & extraction ----------
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from PDF, TXT, CSV. Raise ValueError for unsupported formats."""
        file_name = os.path.basename(file_path).lower()
        with open(file_path, 'rb') as f:
            file_content = f.read()

        if file_name.endswith(".pdf"):
            # Use fitz (PyMuPDF)
            doc = fitz.open(stream=file_content, filetype="pdf")
            texts = []
            for page in doc:
                texts.append(page.get_text("text") or "")
            return "\n".join(texts).strip()

        elif file_name.endswith(".txt"):
            return file_content.decode("utf-8", errors="ignore").strip()

        elif file_name.endswith(".csv"):
            # Use pandas to read CSV from bytes
            df = pd.read_csv(io.BytesIO(file_content))
            return df.to_string(index=False).strip()

        else:
            raise ValueError(f"Unsupported file format: {file_name}")

    # ---------- Text utilities ----------
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()

    def chunk_text_sentences(self, text: str) -> List[str]:
        """Split text into sentence-aware chunks using spaCy + RecursiveCharacterTextSplitter."""
        if not text or not text.strip():
            return []
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        joined = " ".join(sentences)
        chunks = self.text_splitter.split_text(joined)
        # remove empty or whitespace-only chunks
        return [c.strip() for c in chunks if c and c.strip()]

    # ---------- Embeddings & FAISS ----------
    def generate_embeddings(self, text_chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks. Returns float32 numpy array."""
        if not text_chunks:
            raise ValueError("No text chunks provided.")
        # ensure model on device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # sentence-transformers encode arguments
            embeddings = self.sentence_transformer.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
        except TypeError:
            # fallback older API
            embeddings = np.array([self.sentence_transformer.encode(t) for t in text_chunks])
        # ensure float32 for faiss compatibility
        embeddings = np.asarray(embeddings, dtype=np.float32)
        return embeddings

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Create FAISS IndexFlatL2 and add embeddings. Returns the index."""
        if embeddings.size == 0:
            raise ValueError("Embeddings array is empty.")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array (num_examples, dim).")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def create_langchain_vectorstore(self, text_chunks: List[str]) -> LangchainFAISS:
        """Create LangChain FAISS vectorstore from raw texts using HuggingFaceEmbeddings wrapper."""
        # LangchainFAISS.from_texts expects a list of strings and an embeddings object
        return LangchainFAISS.from_texts(texts=text_chunks, embedding=self.lc_embeddings)

    # ---------- Retrieval & reranking ----------
    def retrieve_relevant_context(self, query: str, index: faiss.IndexFlatL2, embeddings: np.ndarray, text_chunks: List[str], top_k: int = 5) -> List[str]:
        """Retrieve and rerank top_k relevant chunks for query using FAISS + cross-encoder."""
        if index.ntotal == 0:
            return []

        # encode query
        q_emb = self.sentence_transformer.encode([query], convert_to_numpy=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        k = min(top_k, max(1, index.ntotal))
        distances, indices = index.search(q_emb, k)
        # indices shape: (1, k)
        idxs = [int(i) for i in indices[0] if i != -1]
        relevant_chunks = [text_chunks[i] for i in idxs]

        if not relevant_chunks:
            return []

        # cross-encoder scoring
        pairs = [(query, chunk) for chunk in relevant_chunks]
        try:
            scores = self.cross_encoder.predict(pairs)
        except Exception:
            # If cross_encoder fails, fall back to cosine similarity on SBERT embeddings
            chunk_embs = self.sentence_transformer.encode(relevant_chunks, convert_to_numpy=True)
            scores = cosine_similarity(q_emb, chunk_embs)[0]

        # sort by score descending and return top 3 (or fewer)
        scored = sorted(zip(scores, relevant_chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored][:3]

    # ---------- Q generation & paraphrase ----------
    def generate_questions(self, text: str, num_questions: int = 5) -> List[str]:
        """Use FLAN-T5 pipeline to generate questions. Returns list of strings."""
        if not text or not text.strip():
            return []
        prompt = f"Generate {num_questions} diverse, specific questions based on: {text}"
        # control max_length; pipeline returns list of dicts
        results = self.qa_model(prompt, max_length=128, num_beams=max(2, num_questions + 2), num_return_sequences=num_questions)
        questions = []
        for r in results:
            # pipeline output can be 'generated_text' or 'text' depending on transformers version
            q = r.get("generated_text") or r.get("text") or ""
            questions.append(q.strip())
        # deduplicate while preserving order
        seen = set()
        dedup = []
        for q in questions:
            if q and q not in seen:
                dedup.append(q)
                seen.add(q)
        return dedup

    def paraphrase_question(self, question: str) -> str:
        """Paraphrase a single question using T5 paraphrase model."""
        if not question:
            return ""
        input_text = f"paraphrase: {question}"
        inputs = self.paraphrase_tokenizer(input_text, return_tensors="pt", truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        outputs = self.paraphrase_model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
        return self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ---------- Augmentation & filtering ----------
    def augment_question(self, question: str, context: str, augmentation_factor: int = 3) -> List[str]:
        """Augment single question using synonyms and a paraphrase. Returns list of unique variants."""
        augmented_questions = {question}
        words = question.split()
        replaceable = [(i, w) for i, w in enumerate(words) if w.lower() in self.synonyms]

        for _ in range(augmentation_factor):
            if not replaceable:
                break
            new_words = words.copy()
            pos, word = random.choice(replaceable)
            # compute best synonym w.r.t context
            context_emb = self.sentence_transformer.encode([context], convert_to_numpy=True)
            syn_list = self.synonyms[word.lower()]
            syn_embs = self.sentence_transformer.encode(syn_list, convert_to_numpy=True)
            sims = cosine_similarity(np.asarray(context_emb, dtype=np.float32), np.asarray(syn_embs, dtype=np.float32))[0]
            best_index = int(np.argmax(sims))
            best_syn = syn_list[best_index]
            new_words[pos] = best_syn
            augmented_questions.add(" ".join(new_words))

        # add paraphrase (best-effort)
        try:
            paraphrased = self.paraphrase_question(question)
            if paraphrased:
                augmented_questions.add(paraphrased)
        except Exception:
            # ignore paraphrase failures
            pass

        return list(augmented_questions)

    def filter_similar_questions(self, questions: List[str], threshold: float = 0.75) -> List[str]:
        """Filter out similar questions using cosine similarity on SBERT embeddings."""
        if not questions:
            return []
        emb = self.sentence_transformer.encode(questions, convert_to_numpy=True)
        emb = np.asarray(emb, dtype=np.float32)
        sim_matrix = cosine_similarity(emb)
        filtered = []
        for i, q in enumerate(questions):
            # allow if it's dissimilar to all previous kept questions
            keep = True
            for j, kept in enumerate(filtered):
                # find index of kept in original list (inefficient but small lists)
                idx_kept = questions.index(kept)
                if sim_matrix[i, idx_kept] >= threshold:
                    keep = False
                    break
            if keep:
                filtered.append(q)
        return filtered

    # ---------- Dataset construction ----------
    def create_qa_dataset(self, text_chunks: List[str], embeddings: np.ndarray, faiss_index: faiss.IndexFlatL2, use_rag: bool = True, in_context_learning: bool = True) -> List[dict]:
        """Create QA dataset from chunks."""
        dataset: List[dict] = []
        vectorstore = self.create_langchain_vectorstore(text_chunks) if in_context_learning else None
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) if in_context_learning and vectorstore is not None else None

        for chunk in text_chunks:
            questions = self.generate_questions(chunk, num_questions=5)
            questions = self.filter_similar_questions(questions)

            for q in questions:
                augmented_qs = self.augment_question(q, chunk)
                for aug_q in augmented_qs:
                    if use_rag:
                        if in_context_learning and retriever:
                            try:
                                docs = retriever.get_relevant_documents(aug_q)
                                context = " ".join([getattr(d, "page_content", str(d)) for d in docs])
                            except Exception:
                                # fallback to FAISS-based retrieval
                                context = " ".join(self.retrieve_relevant_context(aug_q, faiss_index, embeddings, text_chunks))
                        else:
                            context = " ".join(self.retrieve_relevant_context(aug_q, faiss_index, embeddings, text_chunks))
                    else:
                        context = chunk

                    context = self.post_process(context)
                    dataset.append({"role": "user", "content": aug_q})
                    dataset.append({"role": "assistant", "content": context})

        return dataset

    def post_process(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', (text or "")).strip()
        if not text:
            return ""
        text = text[0].upper() + text[1:] if len(text) > 0 else text
        if not text.endswith(('.', '!', '?')):
            text += '.'
        return text

    # ---------- Orchestration ----------
    def process_files(self, file_paths: List[str], chunk_size: int = 1500, use_rag: bool = True, similarity_threshold: float = 0.75, in_context_learning: bool = True) -> Tuple[Optional[str], Optional[List[str]], Optional[List[dict]]]:
        """End-to-end processing for multiple files."""
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

# ---------- Main ----------
def main():
    print("📋 Dataset Creator for Local Environment")
    print("ℹ️ Provide file paths as command-line arguments (e.g., python dataset_creator_fixed.py file1.pdf file2.txt)")

    if len(sys.argv) < 2:
        print("❌ Error: No files provided. Please specify at least one file path.")
        print("Example: python dataset_creator_fixed.py sample.pdf sample.txt")
        sys.exit(1)

    file_paths = sys.argv[1:]
    invalid_files = [f for f in file_paths if not os.path.exists(f)]
    if invalid_files:
        print(f"❌ Error: The following files do not exist: {invalid_files}")
        sys.exit(1)

    # Settings
    chunk_size = 1500
    use_rag = True
    in_context_learning = True
    similarity_threshold = 0.75

    print("🚀 Starting processing pipeline...")
    start_time = time.time()

    creator = DatasetCreator()
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

    output_file = "dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"✅ Dataset saved to '{output_file}'")

    processing_time = time.time() - start_time

    print("\n" + "="*50)
    print(f"🎉 Processing completed in {processing_time:.2f} seconds!")
    print(f"📊 Number of text chunks: {len(text_chunks)}")
    print(f"❓ Number of QA pairs: {len(dataset) // 2}")
    print("="*50 + "\n")

    # Show sample Q&A
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
