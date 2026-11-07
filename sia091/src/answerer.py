# Find the answer to a question inside the chunks found by the retriever.

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

DEFAULT_QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"
MAX_ANSWER_TOKENS = 30


class Answerer:
    # Extractive question answering: the answer is copied from the text

    def __init__(self, model_name=DEFAULT_QA_MODEL):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()

    def answer(self, question, chunks):
        # Try every retrieved chunk and keep the answer with the highest score
        if not question or not question.strip():
            raise ValueError("The question is empty.")
        if not chunks:
            raise ValueError("There are no chunks to answer from.")

        best = None
        for chunk in chunks:
            found = self._answer_one(question, chunk["text"])
            if found is None:
                continue
            if best is None or found["score"] > best["score"]:
                best = {
                    "answer": found["answer"],
                    "score": found["score"],
                    "page": chunk["page"],
                    "context": chunk["text"],
                }

        if best is None or not best["answer"].strip():
            return {
                "answer": "No answer was found in the document.",
                "score": 0.0,
                "page": None,
                "context": "",
            }
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
        lowest = torch.finfo(outputs.start_logits.dtype).min
        start_logits = outputs.start_logits[0].masked_fill(~in_context, lowest)
        end_logits = outputs.end_logits[0].masked_fill(~in_context, lowest)

        start_probs = torch.softmax(start_logits, dim=0)
        end_probs = torch.softmax(end_logits, dim=0)

        # Score every (start, end) pair, keep end >= start and short answers
        scores = start_probs[:, None] * end_probs[None, :]
        scores = torch.triu(torch.tril(scores, diagonal=MAX_ANSWER_TOKENS - 1))

        best_index = int(scores.argmax())
        start_token = best_index // scores.shape[1]
        end_token = best_index % scores.shape[1]

        start_char = int(offsets[start_token][0])
        end_char = int(offsets[end_token][1])
        return {
            "answer": context[start_char:end_char].strip(),
            "score": float(scores.max()),
        }
