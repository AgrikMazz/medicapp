import os
import uuid
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
import fitz
import jsonlines
import hashlib

class Chunker:
    def __init__(self, store_file="data/chunked_output.jsonl"):
        self.store_file = store_file

    def pdf_to_pages(self, path):
        """Extract page text with stable source metadata instead of flattening the PDF."""
        if not os.path.isfile(path) or not path.lower().endswith(".pdf"):
            raise ValueError("Input must be a .pdf file.")

        pages = []
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc):
                pages.append({
                    "page": page_index + 1,
                    "text": page.get_text("text"),
                })
        return pages

    def pdf_to_text(self, path):
        return "\n\n".join(page["text"] for page in self.pdf_to_pages(path))

    def write_chunks(self, chunks, append=False):
        parent = os.path.dirname(self.store_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        mode = "a" if append else "w"
        with jsonlines.open(self.store_file, mode) as writer:
            writer.write_all(chunks)

    def segment_into_chunks(self, text, chunk_size=400, overlap=40, source_id="document", persist=True):
        """Split text or page records while retaining the originating page and offsets.

        Page records use the shape returned by ``pdf_to_pages``. Plain strings remain
        supported for callers that do not have page-level metadata.
        """
        if isinstance(text, str):
            pages = [{"page": None, "text": text}]
        else:
            pages = list(text or [])

        chunks = []
        for page_record in pages:
            page_text = page_record.get("text", "")
            if not page_text or not page_text.strip():
                continue

            page_number = page_record.get("page")
            sentences = sent_tokenize(page_text)
            sentence_records = []
            cursor = 0
            for sentence in sentences:
                start = page_text.find(sentence, cursor)
                if start < 0:
                    start = cursor
                end = min(len(page_text), start + len(sentence))
                cursor = end
                sentence_records.append({"text": sentence, "start": start, "end": end})

            current = []
            current_words = 0

            def emit(records):
                if not records:
                    return
                body = " ".join(record["text"] for record in records).strip()
                if not body:
                    return
                char_start = records[0]["start"]
                char_end = records[-1]["end"]
                stable_id = hashlib.sha256(
                    f"{source_id}:{page_number}:{char_start}:{char_end}:{body}".encode("utf-8")
                ).hexdigest()[:24]
                chunks.append({
                    "id": stable_id,
                    "doc_id": source_id,
                    "text": body,
                    "page_start": page_number,
                    "page_end": page_number,
                    "source_pages": [page_number] if page_number is not None else [],
                    "char_start": char_start,
                    "char_end": char_end,
                    "word_count": len(body.split()),
                })

            for sentence_record in sentence_records:
                sentence_words = len(sentence_record["text"].split())
                if current and current_words + sentence_words > chunk_size:
                    emit(current)

                    overlap_records = []
                    overlap_words = 0
                    if overlap > 0:
                        for prior in reversed(current):
                            overlap_records.insert(0, prior)
                            overlap_words += len(prior["text"].split())
                            if overlap_words >= overlap:
                                break
                    current = overlap_records
                    current_words = overlap_words

                current.append(sentence_record)
                current_words += sentence_words

            emit(current)

        if persist:
            self.write_chunks(chunks)
        return chunks
