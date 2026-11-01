"""
Document loading and semantic chunking module.

Chunking Strategy:
------------------
We use 'Section-Based Semantic Chunking with Context Enrichment':
1. Corporate policy documents are inherently structured into numbered, self-contained sections.
2. Naive character/token-based sliding windows arbitrarily break sentences and policy rules,
   destroying context and weakening vector search precision.
3. Each extracted chunk corresponds to a single policy section and is enriched with the parent
   document's metadata (Document Title, Document ID, Version, and Effective Date).
4. This ensures that the vector embedding encapsulates both the specific clause and the
   governing policy domain, preventing ambiguity between similar rules across different documents.
"""

import re
from pathlib import Path
from typing import List, Dict, Any
from config import KNOWLEDGE_BASE_DIR


def parse_document_header(content: str) -> Dict[str, str]:
    """
    Extract document header metadata:
    - Organization
    - Title
    - Document ID
    - Version
    - Effective date
    """
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    metadata = {
        "organization": "Nexus Workplace Services",
        "doc_title": "Corporate Policy",
        "doc_id": "UNKNOWN",
        "version": "1.0",
        "effective_date": "N/A"
    }

    if len(lines) >= 1:
        metadata["organization"] = lines[0]
    if len(lines) >= 2:
        metadata["doc_title"] = lines[1]
    if len(lines) >= 3:
        header_line = lines[2]
        # Match "Document ID: FIN-EXP-006 | Version: 1.4 | Effective date: 1 October 2026"
        id_match = re.search(r'Document ID:\s*([^|]+)', header_line, re.IGNORECASE)
        version_match = re.search(r'Version:\s*([^|]+)', header_line, re.IGNORECASE)
        date_match = re.search(r'Effective date:\s*(.+)$', header_line, re.IGNORECASE)

        if id_match:
            metadata["doc_id"] = id_match.group(1).strip()
        if version_match:
            metadata["version"] = version_match.group(1).strip()
        if date_match:
            metadata["effective_date"] = date_match.group(1).strip()

    return metadata


def chunk_document_by_sections(file_path: Path) -> List[Dict[str, Any]]:
    """
    Splits a policy document into semantic section chunks.
    Enriches each chunk with document metadata for embedding relevance.
    """
    text = file_path.read_text(encoding="utf-8")
    header_meta = parse_document_header(text)

    # Pattern to detect section headings like:
    # 1. ELIGIBLE EXPENSES
    # 2. NON-ELIGIBLE EXPENSES
    section_pattern = re.compile(r'(?m)^([0-9]+)\.\s+([A-Z\s\-_/&]+)$')
    matches = list(section_pattern.finditer(text))

    chunks = []

    if not matches:
        # Fallback if no numbered sections found: treat as single chunk
        chunk_id = f"{header_meta['doc_id']}_full"
        chunks.append({
            "id": chunk_id,
            "text": text.strip(),
            "metadata": {
                "source_file": file_path.name,
                "doc_title": header_meta["doc_title"],
                "doc_id": header_meta["doc_id"],
                "version": header_meta["version"],
                "effective_date": header_meta["effective_date"],
                "section_num": "0",
                "section_title": "Full Document"
            }
        })
        return chunks

    for i, match in enumerate(matches):
        sec_num = match.group(1).strip()
        sec_title = match.group(2).strip()
        start_idx = match.start()

        # End of this section is start of next section, or end of document
        if i + 1 < len(matches):
            end_idx = matches[i + 1].start()
        else:
            end_idx = len(text)

        sec_body = text[start_idx:end_idx].strip()

        # Enriched text representation: prepends document context to the section text
        # This gives the embedding model vital context about the policy and document id
        enriched_text = (
            f"Policy: {header_meta['doc_title']} (Doc ID: {header_meta['doc_id']})\n"
            f"Effective: {header_meta['effective_date']} | Version: {header_meta['version']}\n\n"
            f"{sec_body}"
        )

        chunk_id = f"{header_meta['doc_id']}_sec_{sec_num}"

        chunks.append({
            "id": chunk_id,
            "text": enriched_text,
            "raw_body": sec_body,
            "metadata": {
                "source_file": file_path.name,
                "doc_title": header_meta["doc_title"],
                "doc_id": header_meta["doc_id"],
                "version": header_meta["version"],
                "effective_date": header_meta["effective_date"],
                "section_num": sec_num,
                "section_title": f"{sec_num}. {sec_title}"
            }
        })

    return chunks


def load_and_chunk_all_documents(knowledge_dir: Path = KNOWLEDGE_BASE_DIR) -> List[Dict[str, Any]]:
    """
    Loads all .txt documents in the knowledge base directory and chunks them.
    """
    all_chunks = []
    txt_files = sorted(knowledge_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt documents found in {knowledge_dir}")

    for file_path in txt_files:
        chunks = chunk_document_by_sections(file_path)
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    # Quick self-test of chunking
    chunks = load_and_chunk_all_documents()
    print(f"Loaded {len(chunks)} chunks across all documents:")
    for c in chunks:
        print(f"- [{c['id']}] {c['metadata']['doc_title']} -> {c['metadata']['section_title']}")

