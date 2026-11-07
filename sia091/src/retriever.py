# Store chunk vectors in a FAISS index and search them with a question.

import faiss


class Retriever:
    # Build a FAISS index from chunks and return the best matches for a question

    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.chunks = []

    def build(self, chunks):
        # Embed all chunks and store them in a FAISS inner-product index
        if not chunks:
            raise ValueError("There are no chunks to index.")
        vectors = self.embedder.encode([chunk["text"] for chunk in chunks])
        # Vectors are normalised, so inner product works like cosine similarity
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks

    def search(self, question, top_k=3):
        # Return the top_k chunks most similar to the question, with scores
        if self.index is None:
            raise RuntimeError("Build the index with a document before searching.")
        if not question or not question.strip():
            raise ValueError("The question is empty.")

        # Never ask for more chunks than the document has
        top_k = max(1, min(top_k, len(self.chunks)))
        query_vector = self.embedder.encode([question])
        scores, positions = self.index.search(query_vector, top_k)

        results = []
        for score, position in zip(scores[0], positions[0]):
            chunk = dict(self.chunks[position])
            chunk["score"] = float(score)
            results.append(chunk)
        return results
