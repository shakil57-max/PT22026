# Find the answer to a question inside the chunks found by the retriever.

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

DEFAULT_QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"


class Answerer:
    # Extractive question answering: the answer is copied from the text

    def __init__(self, model_name=DEFAULT_QA_MODEL, max_answer_tokens=15, scoring="logit"):
        # scoring "prob" compares probabilities inside each chunk
        # scoring "logit" compares raw model scores, which is fairer across chunks
        if scoring not in ("prob", "logit"):
            raise ValueError("scoring must be 'prob' or 'logit'.")
        self.model_name = model_name
        self.max_answer_tokens = max_answer_tokens
        self.scoring = scoring
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()

    def answer(self, question, chunks):
        # Try every retrieved chunk and keep the answer with the best score
        if not question or not question.strip():
            raise ValueError("The question is empty.")
        if not chunks:
            raise ValueError("There are no chunks to answer from.")

        best = None
        for chunk in chunks:
            found = self._answer_one(question, chunk["text"])
            if found is None:
                continue
            if best is None or found[self.scoring] > best[self.scoring]:
                best = dict(found)
                best["page"] = chunk["page"]
                best["context"] = chunk["text"]

        if best is None or not best["answer"].strip():
            return {
                "answer": "No answer was found in the document.",
                "score": 0.0,
                "page": None,
                "context": "",
            }
        # The score shown in the app is always the probability, between 0 and 1
        best["score"] = best["prob"]
        return best

    def _answer_one(self, question, context):
        # Run the model on one chunk and return the best answer span
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=384,
            return_offsets_mapping=True,
        )
        offsets = inputs.pop("offset_mapping")[0]
        sequence_ids = inputs.sequence_ids(0)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Only tokens that belong to the context can be part of an answer
        in_context = torch.tensor([sid == 1 for sid in sequence_ids])
        start_logits = outputs.start_logits[0].masked_fill(~in_context, -1e9)
        end_logits = outputs.end_logits[0].masked_fill(~in_context, -1e9)

        # A valid span ends at or after its start and is not too long
        size = start_logits.shape[0]
        band = torch.triu(torch.tril(torch.ones(size, size), diagonal=self.max_answer_tokens - 1)).bool()

        # Probability score: softmax inside this chunk only
        start_probs = torch.softmax(start_logits, dim=0)
        end_probs = torch.softmax(end_logits, dim=0)
        prob_spans = (start_probs[:, None] * end_probs[None, :]).masked_fill(~band, 0.0)

        # Logit score: raw start + end score, comparable between chunks
        logit_spans = (start_logits[:, None] + end_logits[None, :]).masked_fill(~band, -1e18)

        # The best span is chosen with the scoring method set for this answerer
        spans = logit_spans if self.scoring == "logit" else prob_spans
        best_index = int(spans.argmax())
        start_token = best_index // size
        end_token = best_index % size

        start_char = int(offsets[start_token][0])
        end_char = int(offsets[end_token][1])
        return {
            "answer": context[start_char:end_char].strip(),
            "prob": float(prob_spans[start_token, end_token]),
            "logit": float(logit_spans[start_token, end_token]),
        }

    def candidates(self, question, chunks):
        # Return one answer candidate per chunk, keeping the retrieval score
        # The evaluation script uses this to compare different ranking rules
        if not question or not question.strip():
            raise ValueError("The question is empty.")
        results = []
        for chunk in chunks:
            found = self._answer_one(question, chunk["text"])
            found["page"] = chunk["page"]
            found["retrieval"] = chunk.get("score", 0.0)
            results.append(found)
        return results
