from typing import Dict, List, Any

def chunk_pages(
    doc_id: str,
    doc_name: str,
    pages: List[Dict[str, Any]],
    target_words: int = 300,
    overlap_words: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits document pages into chunks while maintaining provenance.
    Each returned chunk dict contains:
    {
      "chunk_id": "doc_id_chunk_1",
      "document_id": doc_id,
      "document_name": doc_name,
      "page": page_number,
      "section": section_name,
      "text": chunk_text,
      "word_count": count
    }
    """
    chunks: List[Dict[str, Any]] = []
    global_chunk_idx = 1

    for page_info in pages:
        page_num = page_info.get("page", 1)
        section_name = page_info.get("section", "General")
        text = page_info.get("text", "").strip()

        if not text:
            continue

        words = text.split()
        if len(words) <= target_words:
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{global_chunk_idx}",
                "document_id": doc_id,
                "document_name": doc_name,
                "page": page_num,
                "section": section_name,
                "text": text,
                "word_count": len(words)
            })
            global_chunk_idx += 1
        else:
            # Sliding window over words
            start = 0
            while start < len(words):
                end = min(start + target_words, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)

                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{global_chunk_idx}",
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "page": page_num,
                    "section": section_name,
                    "text": chunk_text,
                    "word_count": len(chunk_words)
                })
                global_chunk_idx += 1

                if end >= len(words):
                    break
                start += (target_words - overlap_words)

    return chunks
