import os
import uuid
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
import fitz
import jsonlines

pdf_path="docs\medbook1\medical_book.pdf"

class Chunker:
    def __init__(self, store_file="data/chunked_output.jsonl"):
        self.store_file = store_file

    def pdf_to_text(self, path):
        text = ""

        if os.path.isfile(path) and path.lower().endswith(".pdf"):
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
        elif os.path.isdir(path):
            pdf_files = [f for f in os.listdir(path) if f.lower().endswith(".pdf")]
            pdf_files.sort()
            for fname in pdf_files:
                full_path = os.path.join(path, fname)
                with fitz.open(full_path) as doc:
                    for page in doc:
                        text += page.get_text()
        else:
            raise ValueError("Input must be a .pdf file or a directory containing .pdf files.")

        return text

    def segment_into_chunks(self, text, chunk_size=400, overlap=40):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_length = len(words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                id = uuid.uuid4().hex
                chunks.append({"id" : id, "text" : " ".join(current_chunk)})
                
                overlap_words = []
                overlap_length = 0
                while current_chunk and overlap_length < overlap:
                    s = current_chunk.pop()
                    s_words = s.split()
                    overlap_words.extend(s_words)
                    overlap_length += len(s_words)
                
                overlap_words = overlap_words[::-1]
                current_chunk = [" ".join(overlap_words)]
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            id = uuid.uuid4().hex
            chunks.append({"id" : id, "text" : " ".join(current_chunk)})
            
        with jsonlines.open(self.store_file, "w") as f:
            for chunk in chunks:
                f.write(chunk)
        
        return chunks
