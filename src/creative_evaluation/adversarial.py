"""
Adversarial question generation utilities.
Provides simple, deterministic adversarial transformations suitable for
automatic evaluation and stress-testing of the RAG pipeline.

These functions are intentionally local (no external LLM calls) so they
work offline and are easy to extend to use a paraphrasing model later.
"""
from typing import List
import random
import re


class AdversarialQuestionGenerator:
    """Generate adversarial and paraphrased questions.

    Current implementations are lightweight and rule-based. They are
    intended as a starting point and can be swapped for model-based
    paraphrasers or prompt-based LLM generators later.
    """

    def __init__(self, random_seed: int = 42):
        self.random = random.Random(random_seed)

    def negate_question(self, question: str) -> str:
        """Produce a negated form of a simple question.

        This is a heuristic transformation and works best for simple
        factoid questions.
        """
        # Simple heuristics: insert 'not' or replace 'is' with 'is not' etc.
        if " is " in question:
            return question.replace(" is ", " is not ")
        if " are " in question:
            return question.replace(" are ", " are not ")
        # If no obvious verb, prepend a negation clause
        return "Which of the following is NOT: " + question

    def make_ambiguous(self, question: str) -> str:
        """Make question ambiguous by removing important disambiguators.

        Example: "When was Apple founded?" -> "When was it founded?"
        """
        # Replace proper nouns (capitalized words) with pronouns
        ambiguous = re.sub(r"\b([A-Z][a-z]{2,})\b", "it", question)
        return ambiguous

    def make_multi_hop(self, question: str) -> str:
        """Transform a single-hop question into a multi-hop style by
        expanding it into a chain-of-thought style multi-part question.

        This produces a multi-hop variant that requires combining facts.
        """
        return f"First, find the main subject mentioned in: '{question}'. Then, using that subject, answer: {question}"

    def make_unanswerable(self, question: str) -> str:
        """Make an unanswerable variant by asking about made-up properties."""
        return question + " What color was its imaginary flag?"

    def paraphrase(self, question: str, n: int = 3) -> List[str]:
        """Produce simple paraphrases by reordering or substituting phrases.

        These are intentionally simple; for stronger paraphrasing replace
        this with an LLM or a paraphrase model later.
        """
        variants = set()

        for _ in range(max(4, n * 2)):
            q = question
            # Randomly reorder prepositional phrases
            parts = re.split(r"(,|\.|;)", q)
            if len(parts) > 1 and self.random.random() > 0.5:
                self.random.shuffle(parts)
                q = "".join(parts)

            # Replace common synonyms
            q = re.sub(r"\bwhat is\b", "what's", q, flags=re.I)
            q = re.sub(r"\bwho is\b", "who's", q, flags=re.I)
            q = re.sub(r"\bwhen was\b", "when did\b", q, flags=re.I)

            # Minor punctuation changes
            if self.random.random() > 0.7:
                q = q.rstrip('?') + ' ?'

            variants.add(q.strip())
            if len(variants) >= n:
                break

        return list(variants)

    def generate_adversarial_set(self, question: str) -> List[dict]:
        """Return a list of adversarial variants with type metadata."""
        out = []
        out.append({"type": "original", "q": question})
        out.append({"type": "negated", "q": self.negate_question(question)})
        out.append({"type": "ambiguous", "q": self.make_ambiguous(question)})
        out.append({"type": "multi_hop", "q": self.make_multi_hop(question)})
        out.append({"type": "unanswerable", "q": self.make_unanswerable(question)})
        paras = self.paraphrase(question, n=3)
        for i, p in enumerate(paras, 1):
            out.append({"type": f"paraphrase_{i}", "q": p})

        return out


if __name__ == "__main__":
    g = AdversarialQuestionGenerator()
    q = "When was the Declaration of Independence signed?"
    for item in g.generate_adversarial_set(q):
        print(item)

