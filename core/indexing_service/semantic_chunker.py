import logging
import re
from typing import List, Optional
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import torch
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data with error handling
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")
    raise RuntimeError(f"NLTK data download failed: {e}")

# Initialize SentenceTransformer model
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    logger.info(f"SentenceTransformer initialized on device: {model.device}")
except Exception as e:
    logger.error(f"Failed to initialize SentenceTransformer: {e}")
    raise RuntimeError(f"SentenceTransformer initialization failed: {e}")

def get_optimal_batch_size(device: str) -> int:
    """
    Determine optimal batch size based on device and GPU memory.

    Args:
        device (str): Device type ('cuda' or 'cpu').

    Returns:
        int: Optimal batch size for embedding generation.
    """
    if device.startswith("cuda"):
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            return min(32, gpu_memory // (768 * 4))  # 768-dim embeddings
        except Exception as e:
            logger.warning(f"Failed to get GPU memory, defaulting to batch size 16: {e}")
            return 16
    return 16

batch_size = get_optimal_batch_size(device)

def semantic_chunking(
    text_documents: List[Document],
    table_documents: List[Document],
    initial_chunk_size: int = 3,
    min_chunk_size: int = 180,
    max_chunk_size: int = 500,
    overlap_sentences: int = 1,
    batch_size: int = batch_size,
    max_segment_sentences: int = 1000,
    base_merge_threshold: float = 0.8,
    base_relaxed_threshold: float = 0.6
) -> List[Document]:
    """
    Perform semantic chunking on two separate document arrays: text and table.

    Text documents are chunked semantically using embeddings. Table documents are passed
    through unchanged with updated chunk IDs. The max_chunk_size is treated as a soft limit,
    stopping at the nearest semantic boundary.

    Args:
        text_documents: List of text Document objects.
        table_documents: List of table Document objects.
        initial_chunk_size: Number of sentences in each initial text chunk.
        min_chunk_size: Minimum word count for a text chunk.
        max_chunk_size: Soft maximum word count per text chunk (~500 words).
        overlap_sentences: Number of sentences to overlap between text chunks.
        batch_size: Batch size for embedding generation.
        max_segment_sentences: Max sentences per segment for large text documents.
        base_merge_threshold: Base similarity threshold for merging text chunks.
        base_relaxed_threshold: Base relaxed threshold for small text chunks.

    Returns:
        List[Document]: List of Document objects containing chunks and metadata.
    """
    if not any([text_documents, table_documents]):
        logger.warning("No documents provided for chunking")
        return []

    # Log input document sources
    text_sources = {doc.metadata.get("source", "unknown") for doc in text_documents}
    table_sources = {doc.metadata.get("source", "unknown") for doc in table_documents}
    logger.debug(f"Text document sources: {text_sources}")
    logger.debug(f"Table document sources: {table_sources}")

    # Collect all documents (text and tables)
    output_documents = []

    # Pass through table documents
    for doc in table_documents:
        content_type = doc.metadata.get("content_type", "table")
        if content_type != "table":
            logger.warning(f"Unexpected content_type '{content_type}' in table document")
            continue
        output_documents.append(Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "chunk_id": f"table_{len(output_documents)}",
                "parent_doc_id": doc.metadata.get("doc_id", "unknown")
            }
        ))

    # Process text documents directly
    for doc in text_documents:
        content_type = doc.metadata.get("content_type", "text")
        if content_type != "text":
            logger.warning(f"Unexpected content_type '{content_type}' in text document")
            continue

        try:
            output_documents.extend(_chunk_text_document(
                doc=doc,
                initial_chunk_size=initial_chunk_size,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                overlap_sentences=overlap_sentences,
                batch_size=batch_size,
                max_segment_sentences=max_segment_sentences,
                base_merge_threshold=base_merge_threshold,
                base_relaxed_threshold=base_relaxed_threshold
            ))
        except Exception as e:
            logger.error(f"Chunking failed for source {doc.metadata.get('source', 'unknown')}: {e}")
            output_documents.append(Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "chunk_id": f"error_{len(output_documents)}",
                    "error": str(e),
                    "parent_doc_id": doc.metadata.get("doc_id", "unknown")
                }
            ))

    chunk_sizes = [len(doc.page_content.split()) for doc in output_documents if doc.metadata.get("content_type") == "text"]
    if chunk_sizes:
        logger.info(f"Generated {len(output_documents)} chunks; Text chunk sizes: min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={np.mean(chunk_sizes):.1f}")
    else:
        logger.info(f"Generated {len(output_documents)} chunks; No text chunks produced")
    return output_documents

def _chunk_text_document(
    doc: Document,
    initial_chunk_size: int,
    min_chunk_size: int,
    max_chunk_size: int,
    overlap_sentences: int,
    batch_size: int,
    max_segment_sentences: int,
    base_merge_threshold: float,
    base_relaxed_threshold: float
) -> List[Document]:
    """
    Chunk a single text Document semantically using sentence embeddings.

    Chunks below min_chunk_size are merged with previous or next chunk based on similarity.
    Chunks respect a soft max_chunk_size (~500 words) by stopping at the nearest semantic boundary.

    Args:
        doc: Input Document object.
        initial_chunk_size: Number of sentences in each initial chunk.
        min_chunk_size: Minimum word count for a chunk (180).
        max_chunk_size: Soft maximum word count per chunk (~500).
        overlap_sentences: Number of sentences to overlap between chunks.
        batch_size: Batch size for embedding generation.
        max_segment_sentences: Max sentences per segment.
        base_merge_threshold: Base similarity threshold for merging.
        base_relaxed_threshold: Base relaxed threshold for small chunks.

    Returns:
        List[Document]: List of Document objects containing chunks.
    """
    text = doc.page_content
    metadata = doc.metadata
    source = metadata.get("source", "unknown")
    page_number = metadata.get("page_number", 0)
    parent_doc_id = metadata.get("doc_id", "unknown")

    # Split text into sentences
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.error(f"Sentence tokenization failed for source {source}, page {page_number}: {e}")
        return [Document(
            page_content=text,
            metadata={
                **metadata,
                "chunk_id": "error_0",
                "error": f"Tokenization failed: {e}",
                "parent_doc_id": parent_doc_id
            }
        )]

    if not sentences:
        logger.warning(f"No sentences found in source {source}, page {page_number}")
        return [Document(
            page_content=text,
            metadata={
                **metadata,
                "chunk_id": "empty_0",
                "parent_doc_id": parent_doc_id
            }
        )]

    # Detect section headers
    def is_section_header(sentence: str, index: int) -> bool:
        sentence = sentence.strip()
        is_header_by_metadata = metadata.get("element_type", "").lower() == "header"
        is_header_by_pattern = (
            len(sentence.split()) < 10 and
            (sentence[0].isupper() or sentence.endswith(':')) and
            not re.match(r'^-', sentence)
        )
        return is_header_by_metadata or is_header_by_pattern

    header_indices = [i for i, s in enumerate(sentences) if is_section_header(s, i)]

    # Segment large documents
    segments = []
    for i in range(0, len(sentences), max_segment_sentences):
        segment_start = i
        segment_end = min(i + max_segment_sentences, len(sentences))
        while segment_end < len(sentences) and segment_end in header_indices:
            segment_end += 1
        segments.append((segment_start, segment_end))

    all_chunks = []
    segment_embeddings = []

    # Process each segment
    logger.debug(f"Chunking with device: {model.device}")
    for seg_start, seg_end in segments:
        seg_sentences = sentences[seg_start:seg_end]
        if not seg_sentences:
            continue

        # Generate embeddings
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                embeddings = executor.submit(
                    model.encode,
                    seg_sentences,
                    show_progress_bar=False,
                    batch_size=batch_size
                ).result()
        except Exception as e:
            logger.error(f"Embedding failed for segment {seg_start}-{seg_end}, source {source}, page {page_number}: {e}")
            return [Document(
                page_content=text,
                metadata={
                    **metadata,
                    "chunk_id": f"error_{len(all_chunks)}",
                    "error": f"Embedding failed: {e}",
                    "parent_doc_id": parent_doc_id
                }
            )]

        segment_embeddings.append(embeddings)

        # Compute adaptive thresholds
        similarities = []
        embeddings_tensor = torch.tensor(embeddings, device=device)
        for i in range(len(embeddings) - 1):
            sim = torch.cosine_similarity(
                embeddings_tensor[i].unsqueeze(0),
                embeddings_tensor[i + 1].unsqueeze(0)
            ).item()
            similarities.append(sim)
        merge_threshold = min(base_merge_threshold, np.percentile(similarities, 75)) if similarities else base_merge_threshold
        relaxed_threshold = min(base_relaxed_threshold - 0.1, np.percentile(similarities, 25)) if similarities else base_relaxed_threshold - 0.1

        # Create initial chunks
        chunk_ranges = []
        for i in range(0, len(seg_sentences), initial_chunk_size):
            start = i
            end = min(i + initial_chunk_size, len(seg_sentences))
            chunk_ranges.append((start, end))

        # Compute average embedding for a range
        def get_avg_emb(start: int, end: int) -> np.ndarray:
            return np.mean([embeddings[j] for j in range(start, end)], axis=0)

        # Count words in a chunk
        def count_words(start: int, end: int) -> int:
            chunk_text = ' '.join(seg_sentences[start:end])
            return len(chunk_text.split())

        # Check if chunk is near a captioned element
        def near_captioned_element(start: int, end: int) -> bool:
            caption = metadata.get("caption", "")
            if caption:
                chunk_text = ' '.join(seg_sentences[start:end]).lower()
                return caption.lower() in chunk_text
            return False

        # Find nearest semantic boundary based on word count
        def find_semantic_boundary(start: int, target_word_count: int) -> int:
            word_count = 0
            end = start
            while end < len(seg_sentences) and word_count < target_word_count * 1.2:
                chunk_text = ' '.join(seg_sentences[start:end + 1])
                word_count = len(chunk_text.split())
                end += 1
            if end >= len(seg_sentences):
                return len(seg_sentences)
            for i in range(max(end - 2, start + 1), min(end + 2, len(seg_sentences))):
                if i + 1 < len(seg_sentences):
                    emb_i = embeddings[i]
                    emb_next = embeddings[i + 1]
                    sim = torch.cosine_similarity(
                        torch.tensor(emb_i, device=device).unsqueeze(0),
                        torch.tensor(emb_next, device=device).unsqueeze(0)
                    ).item()
                    if sim < merge_threshold:
                        chunk_text = ' '.join(seg_sentences[start:i + 1])
                        if len(chunk_text.split()) <= target_word_count * 1.2:
                            return i + 1
            return end

        # Merge similar chunks
        max_iterations = 100
        iteration = 0
        while True:
            iteration += 1
            if iteration > max_iterations:
                logger.error(f"Max iterations reached in merging loop for source {source}, page {page_number}")
                break
            new_chunk_ranges = []
            i = 0
            merged = False
            while i < len(chunk_ranges):
                if i + 1 < len(chunk_ranges):
                    emb_i = get_avg_emb(chunk_ranges[i][0], chunk_ranges[i][1])
                    emb_ip1 = get_avg_emb(chunk_ranges[i+1][0], chunk_ranges[i+1][1])
                    emb_i_tensor = torch.tensor(emb_i, device=device).unsqueeze(0)
                    emb_ip1_tensor = torch.tensor(emb_ip1, device=device).unsqueeze(0)
                    sim = torch.cosine_similarity(emb_i_tensor, emb_ip1_tensor).item()
                    if sim > merge_threshold or near_captioned_element(chunk_ranges[i][0], chunk_ranges[i+1][1]):
                        new_start = chunk_ranges[i][0]
                        new_end = chunk_ranges[i+1][1]
                        word_count = count_words(new_start, new_end)
                        if word_count <= max_chunk_size * 1.2:
                            new_chunk_ranges.append((new_start, new_end))
                            i += 2
                            merged = True
                        else:
                            new_chunk_ranges.append(chunk_ranges[i])
                            i += 1
                    else:
                        new_chunk_ranges.append(chunk_ranges[i])
                        i += 1
                else:
                    new_chunk_ranges.append(chunk_ranges[i])
                    i += 1
            if not merged:
                break
            chunk_ranges = new_chunk_ranges

        # Enforce minimum chunk size by merging small chunks
        final_chunk_ranges = []
        i = 0
        while i < len(chunk_ranges):
            start, end = chunk_ranges[i]
            word_count = count_words(start, end)
            if word_count >= min_chunk_size or near_captioned_element(start, end):
                final_chunk_ranges.append((start, end))
                i += 1
            else:
                # Always merge with a neighbor if possible
                if i > 0:
                    prev_start, prev_end = final_chunk_ranges[-1]
                    combined_start = prev_start
                    combined_end = end
                    combined_word_count = count_words(combined_start, combined_end)
                    if combined_word_count <= max_chunk_size * 1.2:
                        final_chunk_ranges[-1] = (combined_start, combined_end)
                        logger.debug(f"Merged small chunk ({word_count} words) with previous, new size: {combined_word_count}, source={source}, page={page_number}")
                        i += 1
                        continue
                if i + 1 < len(chunk_ranges):
                    next_start, next_end = chunk_ranges[i+1]
                    combined_start = start
                    combined_end = next_end
                    combined_word_count = count_words(combined_start, combined_end)
                    if combined_word_count <= max_chunk_size * 1.2:
                        final_chunk_ranges.append((combined_start, combined_end))
                        logger.debug(f"Merged small chunk ({word_count} words) with next, new size: {combined_word_count}, source={source}, page={page_number}")
                        i += 2
                        continue
                # If no merge is possible, keep the chunk as-is
                final_chunk_ranges.append((start, end))
                logger.warning(f"Kept small chunk ({word_count} words) unmerged, source={source}, page={page_number}")
                i += 1

        # Adjust for global sentence indices
        seg_chunks = [(seg_start + start, seg_start + end) for start, end in final_chunk_ranges]
        all_chunks.extend(seg_chunks)

    # Apply soft maximum chunk size and overlap
    final_chunks = []
    chunk_similarities = []
    for i, (start, end) in enumerate(all_chunks):
        chunk_text = ' '.join(sentences[start:end])
        word_count = len(chunk_text.split())
        if word_count > max_chunk_size:
            k = start
            while k < end:
                sub_start = k
                sub_end = find_semantic_boundary(sub_start, max_chunk_size)
                chunk_text = ' '.join(sentences[sub_start:sub_end])
                overlap_start = max(0, sub_start - overlap_sentences if i > 0 else sub_start)
                final_text = ' '.join(sentences[overlap_start:sub_end])
                final_word_count = len(final_text.split())
                final_chunks.append(Document(
                    page_content=final_text,
                    metadata={
                        **metadata,
                        "chunk_id": f"text_{len(final_chunks)}",
                        "start_sentence": sub_start,
                        "parent_doc_id": parent_doc_id
                    }
                ))
                logger.debug(f"Split large chunk: {final_word_count} words, source={source}")
                k = sub_end
        else:
            overlap_start = max(0, start - overlap_sentences if i > 0 else start)
            final_text = ' '.join(sentences[overlap_start:end])
            final_word_count = len(final_text.split())
            final_chunks.append(Document(
                page_content=final_text,
                metadata={
                    **metadata,
                    "chunk_id": f"text_{len(final_chunks)}",
                    "start_sentence": start,
                    "parent_doc_id": parent_doc_id
                }
            ))
            logger.debug(f"Chunk created: {final_word_count} words, source={source}")

        # Compute chunk similarity for metrics
        if i > 0 and segment_embeddings:
            try:
                prev_seg_idx = next(idx for idx, (s, e) in enumerate(segments) if all_chunks[i-1][0] >= s and all_chunks[i-1][1] <= e)
                curr_seg_idx = next(idx for idx, (s, e) in enumerate(segments) if start >= s and end <= e)
                prev_emb = np.mean(segment_embeddings[prev_seg_idx][all_chunks[i-1][0]-segments[prev_seg_idx][0]:all_chunks[i-1][1]-segments[prev_seg_idx][0]], axis=0)
                curr_emb = np.mean(segment_embeddings[curr_seg_idx][start-segments[curr_seg_idx][0]:end-segments[curr_seg_idx][0]], axis=0)
                prev_emb_tensor = torch.tensor(prev_emb, device=device).unsqueeze(0)
                curr_emb_tensor = torch.tensor(curr_emb, device=device).unsqueeze(0)
                sim = torch.cosine_similarity(prev_emb_tensor, curr_emb_tensor).item()
                chunk_similarities.append(sim)
            except Exception as e:
                logger.warning(f"Failed to compute chunk similarity for source {source}, page {page_number}: {e}")

    # Log metrics
    chunk_sizes = [len(chunk.page_content.split()) for chunk in final_chunks]
    if chunk_sizes:
        logger.info(f"Generated {len(final_chunks)} text chunks; Text chunk sizes: min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={np.mean(chunk_sizes):.1f}")
    if chunk_similarities:
        logger.debug(f"Average chunk similarity for source {source}: {np.mean(chunk_similarities):.2f}")

    return final_chunks

if __name__ == "__main__":
    # Sample documents for testing
    sample_text_docs = [
        Document(
            page_content="APPLICATION MANAGEMENT COMMANDS",
            metadata={
                "source": "advanced-linux-commands-cheat-sheet-red-hat-developer.pdf",
                "content_type": "text",
                "page_number": 1,
                "element_type": "section_header",
                "level": 1,
                "index": 0,
                "doc_id": "sample_doc_1"
            }
        ),
        Document(
            page_content="systemctl [options] <subcommand>",
            metadata={
                "source": "advanced-linux-commands-cheat-sheet-red-hat-developer.pdf",
                "content_type": "text",
                "page_number": 1,
                "element_type": "text",
                "level": 2,
                "index": 1,
                "doc_id": "sample_doc_1"
            }
        ),
        Document(
            page_content="$ systemctl status httpd",
            metadata={
                "source": "advanced-linux-commands-cheat-sheet-red-hat-developer.pdf",
                "content_type": "text",
                "page_number": 1,
                "element_type": "list_item",
                "level": 2,
                "index": 2,
                "doc_id": "sample_doc_1"
            }
        )
    ]
    sample_table_docs = [
        Document(
            page_content=    "| $ sudo dnf history undo last Updating Subscription Management repositories. Last metadata expiration check: 3:47:28 ago on Wed 02 Feb 2022 05:08:48   | $ sudo dnf history undo last Updating Subscription Management repositories. Last metadata expiration check: 3:47:28 ago on Wed 02 Feb 2022 05:08:48                 | $ sudo dnf history undo last Updating Subscription Management repositories. Last metadata expiration check: 3:47:28 ago on Wed 02 Feb 2022 05:08:48   |\n|-------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|\n| Architecture Size ===========================================================================                                                         | Architecture Size ===========================================================================                                                                       | Architecture Size ===========================================================================                                                         |\n| Version                                                                                                                                               | Version                                                                                                                                                             | Version                                                                                                                                               |\n| ================================================                                                                                                      | Removing: dotnet x86_64 0 x86_64 21 M x86_64 13 M x86_64 11 M x86_64 200 k x86_64 345 k x86_64 65 M x86_64 268 M x86_64 26 M x86_64 6.2 M x86_64 1.1 M x86_64 18 M  | 6.0.101-2.el8_5 6.0.1-2.el8_5 6.0.1-2.el8_5 6.0.1-2.el8_5 6.0.1-2.el8_5 6.0.1-2.el8_5 6.0.1-2.el8_5                                                   |\n|                                                                                                                                                       |                                                                                                                                                                     | Removing dependent packages: aspnetcore-runtime-6.0 @rhel-8-for-x86_64-appstream-rpms                                                                 |\n|                                                                                                                                                       | aspnetcore-targeting-pack-6.0 @rhel-8-for-x86_64-appstream-rpms                                                                                                     |                                                                                                                                                       |\n|                                                                                                                                                       | dotnet-apphost-pack-6.0                                                                                                                                             |                                                                                                                                                       |\n|                                                                                                                                                       | @rhel-8-for-x86_64-appstream-rpms dotnet-host                                                                                                                       |                                                                                                                                                       |\n|                                                                                                                                                       | @rhel-8-for-x86_64-appstream-rpms dotnet-hostfxr-6.0                                                                                                                |                                                                                                                                                       |\n|                                                                                                                                                       | @rhel-8-for-x86_64-appstream-rpms dotnet-runtime-6.0                                                                                                                |                                                                                                                                                       |\n|                                                                                                                                                       | @rhel-8-for-x86_64-appstream-rpms dotnet-sdk-6.0                                                                                                                    |                                                                                                                                                       |\n| 6.0.101-2.el8_5                                                                                                                                       | @rhel-8-for-x86_64-appstream-rpms dotnet-targeting-pack-6.0                                                                                                         |                                                                                                                                                       |\n| 6.0.1-2.el8_5                                                                                                                                         | @rhel-8-for-x86_64-appstream-rpms dotnet-templates-6.0                                                                                                              |                                                                                                                                                       |\n| 6.0.101-2.el8_5                                                                                                                                       | @rhel-8-for-x86_64-appstream-rpms                                                                                                                                   |                                                                                                                                                       |\n|                                                                                                                                                       | lttng-ust @rhel-8-for-x86_64-appstream-rpms netstandard-targeting-pack-2.1                                                                                          |                                                                                                                                                       |\n| 2.8.1-11.el8                                                                                                                                          |                                                                                                                                                                     |                                                                                                                                                       |\n|                                                                                                                                                       | @rhel-8-for-x86_64-appstream-rpms                                                                                                                                   |                                                                                                                                                       |\n| 6.0.101-2.el8_5                                                                                                                                       |                                                                                                                                                                     |                                                                                                                                                       |",
            metadata={
                "source": "advanced-linux-commands-cheat-sheet-red-hat-developer.pdf",
                "content_type": "table",
                "page_number": 2,
                "caption": "Table 1: System Commands",
                "doc_id": "sample_doc_2"
            }
        )
    ]
    try:
        chunks = semantic_chunking(
            text_documents=sample_text_docs,
            table_documents=sample_table_docs,
            initial_chunk_size=1,
            min_chunk_size=180,
            max_chunk_size=500,
            overlap_sentences=1,
            batch_size=batch_size
        )
        for i, chunk in enumerate(chunks):
            logger.info(
                f"Chunk {i+1} ({chunk.metadata.get('content_type', 'unknown')}, {len(chunk.page_content.split())} words): {chunk.page_content[:100]}..."
            )
    except Exception as e:
        logger.error(f"Test chunking failed: {e}")