# Split document pages into overlapping chunks for retrieval.


def chunk_pages(pages, chunk_size=120, overlap=30):
    # Turn (page_number, text) pairs into a list of chunk dictionaries
    # chunk_size: number of words in each chunk
    # overlap: words shared between neighbouring chunks, so an answer
    # that falls on a boundary is still found
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1.")

    step = chunk_size - overlap
    chunks = []

    for page_number, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            piece = words[start:start + chunk_size]
            if not piece:
                break
            chunks.append({
                "text": " ".join(piece),
                "page": page_number,
            })
            # Stop once this chunk already reaches the end of the page
            if start + chunk_size >= len(words):
                break

    return chunks
