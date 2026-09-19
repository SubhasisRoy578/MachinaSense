import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "knowledge"))
INDEX_FILE = os.path.join(KNOWLEDGE_DIR, "vector_index.json")
CATALOG_FILE = os.path.join(KNOWLEDGE_DIR, "documents_catalog.json")

class VectorStore:
    """
    Lightweight TF-IDF + Cosine Similarity Vector Store.
    Stores document chunks, metadata, and persistent TF-IDF index.
    """
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}  # doc_id -> doc_meta
        self.chunks: List[Dict[str, Any]] = []          # List of all chunk dicts
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.is_indexed = False
        
        # Ensure knowledge artifacts directory exists
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        self.load()

    def save(self):
        """Persists document catalog and chunk index to disk."""
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        catalog_data = list(self.documents.values())
        with open(CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump(catalog_data, f, indent=2)

        index_data = {
            "method": "TF-IDF + Cosine Similarity Vector Search",
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "chunks": self.chunks
        }
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    def load(self):
        """Loads persistent document catalog and chunk index."""
        if os.path.exists(CATALOG_FILE):
            try:
                with open(CATALOG_FILE, "r", encoding="utf-8") as f:
                    docs_list = json.load(f)
                    self.documents = {d["id"]: d for d in docs_list}
            except Exception as e:
                print(f"Warning loading documents catalog: {e}")

        if os.path.exists(INDEX_FILE):
            try:
                with open(INDEX_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", [])
                    self._rebuild_tfidf_index()
            except Exception as e:
                print(f"Warning loading vector index: {e}")

        # Seed initial technical domain documentation if store is empty
        if not self.documents or not self.chunks:
            self._seed_default_domain_docs()

    def _seed_default_domain_docs(self):
        """Seeds standard C-MAPSS turbofan engineering manuals into knowledge base."""
        from datetime import datetime
        
        doc1_meta = {
            "id": "DOC-901",
            "title": "C-MAPSS FD001 Turbofan Maintenance & Degradation Guide",
            "type": "PDF",
            "size": 14500000,
            "uploadDate": datetime.now().isoformat(),
            "processingStatus": "ready",
            "indexedChunks": 4,
            "sourceCategory": "Manual"
        }
        doc1_chunks = [
            {
                "chunk_id": "DOC-901_chunk_1",
                "document_id": "DOC-901",
                "document_name": "C-MAPSS FD001 Turbofan Maintenance & Degradation Guide",
                "page": 1,
                "section": "Section 1.2 - High-Pressure Compressor Degradation",
                "text": "High-Pressure Compressor (HPC) degradation is the primary failure mode in FD001 turbofan engines. Symptoms include elevated HPC outlet temperature (sensor s3), LPT outlet temperature (sensor s4), and static pressure fluctuations (sensor s11). When HPC outlet temperature s3 exceeds 1600 °R and RUL falls below 30 cycles, immediate inspection of HPC stator vanes and blade tip clearances is required.",
                "word_count": 62
            },
            {
                "chunk_id": "DOC-901_chunk_2",
                "document_id": "DOC-901",
                "document_name": "C-MAPSS FD001 Turbofan Maintenance & Degradation Guide",
                "page": 2,
                "section": "Section 2.4 - Fan & Core Speed Calibration Bounds",
                "text": "Physical and corrected fan speeds (sensors s8 and s13) operate nominally near 2388 RPM. Deviations in corrected core speed (sensor s14 below 8150 RPM) alongside elevated bleed enthalpy (sensor s17 above 394 BTU/lb) indicate thermal degradation and air leakage in HPT cooling seals.",
                "word_count": 45
            },
            {
                "chunk_id": "DOC-901_chunk_3",
                "document_id": "DOC-901",
                "document_name": "C-MAPSS FD001 Turbofan Maintenance & Degradation Guide",
                "page": 3,
                "section": "Section 3.1 - Isolation Forest Anomaly Thresholds",
                "text": "Isolation Forest anomaly scores exceeding 0.60 indicate significant deviation from healthy baseline flight cycles. Anomaly scores greater than 0.75 signal critical acoustic or thermal instability requiring preventive maintenance before RUL drops below 15 cycles.",
                "word_count": 37
            }
        ]

        doc2_meta = {
            "id": "DOC-902",
            "title": "Spindle & Compressor Shaft Bearing Specifications",
            "type": "PDF",
            "size": 2300000,
            "uploadDate": datetime.now().isoformat(),
            "processingStatus": "ready",
            "indexedChunks": 2,
            "sourceCategory": "Specification"
        }
        doc2_chunks = [
            {
                "chunk_id": "DOC-902_chunk_1",
                "document_id": "DOC-902",
                "document_name": "Spindle & Compressor Shaft Bearing Specifications",
                "page": 1,
                "section": "Section 4.2 - Bearing Wear & Acoustic Resonance",
                "text": "High frequency resonance (>2kHz) on compressor shaft bearings often precedes catastrophic spalling by 7 to 14 flight cycles. Elevated LPC outlet temperature (sensor s2 > 645 °R) and bypass duct pressure (sensor s6 > 21.6 psia) confirm thermal expansion in bearing housing assemblies.",
                "word_count": 44
            }
        ]

        doc3_meta = {
            "id": "DOC-903",
            "title": "CNC & Turbofan Fuel System Troubleshooting Manual",
            "type": "DOCX",
            "size": 850000,
            "uploadDate": datetime.now().isoformat(),
            "processingStatus": "ready",
            "indexedChunks": 2,
            "sourceCategory": "Manual"
        }
        doc3_chunks = [
            {
                "chunk_id": "DOC-903_chunk_1",
                "document_id": "DOC-903",
                "document_name": "CNC & Turbofan Fuel System Troubleshooting Manual",
                "page": 1,
                "section": "Troubleshooting Chart B - Fuel Ratio & Bypass Flow",
                "text": "Erratic fuel flow ratio (sensor s12) and bypass ratio fluctuations (sensor s15) indicate metering valve contamination or fuel pump cavitation. Flush fuel supply lines and recalibrate flow control valves during scheduled maintenance.",
                "word_count": 35
            }
        ]

        self.add_document(doc1_meta, doc1_chunks)
        self.add_document(doc2_meta, doc2_chunks)
        self.add_document(doc3_meta, doc3_chunks)


    def _rebuild_tfidf_index(self):
        """Rebuilds the TF-IDF matrix from stored chunks."""
        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            self.is_indexed = False
            return

        corpus = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            stop_words='english'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True

    def add_document(self, doc_meta: Dict[str, Any], doc_chunks: List[Dict[str, Any]]):
        """Adds a document and its chunks to the store and updates index."""
        doc_id = doc_meta["id"]
        self.documents[doc_id] = doc_meta
        
        # Remove old chunks for this doc if re-indexing
        self.chunks = [c for c in self.chunks if c["document_id"] != doc_id]
        self.chunks.extend(doc_chunks)

        self._rebuild_tfidf_index()
        self.save()

    def search(self, query: str, top_k: int = 5, min_score: float = 0.05) -> List[Dict[str, Any]]:
        """
        Performs TF-IDF cosine similarity search for the given query.
        Returns top-K relevant chunks with exact source provenance.
        """
        if not self.is_indexed or not self.vectorizer or self.tfidf_matrix is None or not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "document_name": chunk["document_name"],
                    "page": chunk.get("page", 1),
                    "section": chunk.get("section", "General"),
                    "text": chunk["text"],
                    "excerpt": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
                    "relevance_score": round(score, 4)
                })

        return results

    def get_documents(self) -> List[Dict[str, Any]]:
        return list(self.documents.values())

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get(doc_id)

    def get_chunks_for_document(self, doc_id: str) -> List[Dict[str, Any]]:
        return [c for c in self.chunks if c["document_id"] == doc_id]

vector_store = VectorStore()
