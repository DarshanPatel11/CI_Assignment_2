"""
Question Generation Module

Generates diverse Q&A pairs from Wikipedia corpus for evaluation.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


@dataclass
class QAPair:
    """Represents a question-answer pair for evaluation."""
    question_id: str
    question: str
    answer: str
    source_url: str
    source_title: str
    source_chunk_id: str
    question_type: str  # factual, comparative, inferential, multi-hop
    difficulty: str     # easy, medium, hard

    def to_dict(self) -> dict:
        return asdict(self)


class QuestionGenerator:
    """Generates diverse questions from Wikipedia corpus using LLMs."""

    QUESTION_TYPES = ["factual", "comparative", "inferential", "multi-hop"]
    DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

    # Templates for different question types
    TEMPLATES = {
        "factual": [
            "Generate a factual question that can be answered from this text. The question should ask about specific facts, dates, names, or events mentioned in the passage.",
            "Create a simple factual question based on the following text. Focus on who, what, when, where, or how much.",
            "What is {entity}?",
            "Who is {person}?",
            "When was {event} established/founded?",
            "Where is {location} located?",
            "How does {process} work?"
        ],
        "comparative": [
            "Generate a comparison question based on this text. The question should ask to compare or contrast different elements mentioned.",
            "Create a question that requires comparing two or more things mentioned in the text.",
            "What are the main differences between {concept1} and {concept2}?"
        ],
        "inferential": [
            "Generate an inferential question that requires reasoning beyond the literal text. The answer should be derived through logical inference.",
            "Create a question that requires drawing conclusions from the information provided."
        ],
        "multi-hop": [
            "Generate a complex question that requires connecting multiple pieces of information from the text.",
            "Create a question that needs combining facts from different parts of the passage to answer."
        ]
    }

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        data_dir: str = "data",
        device: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.eval_dir = self.data_dir / "evaluation"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model for question generation
        print(f"Loading question generation model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _generate_question_id(self, chunk_id: str, q_type: str, idx: int) -> str:
        """Generate unique question ID."""
        hash_input = f"{chunk_id}_{q_type}_{idx}"
        return f"q_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"

    def _estimate_difficulty(self, question: str, answer: str) -> str:
        """Estimate question difficulty based on complexity."""
        # Simple heuristics
        question_words = len(question.split())
        answer_words = len(answer.split())

        if question_words < 10 and answer_words < 15:
            return "easy"
        elif question_words > 20 or answer_words > 40:
            return "hard"
        else:
            return "medium"

    def generate_from_chunk(
        self,
        chunk: Dict,
        question_type: str = "factual",
        num_questions: int = 1
    ) -> List[QAPair]:
        """
        Generate Q&A pairs from a single chunk.

        Args:
            chunk: Dict with 'chunk_id', 'content', 'url', 'title'
            question_type: Type of question to generate
            num_questions: Number of questions to generate

        Returns:
            List of QAPair objects
        """
        content = chunk.get('content', '')
        if len(content.split()) < 30:  # Skip very short chunks
            return []

        qa_pairs = []
        template = random.choice(self.TEMPLATES.get(question_type, self.TEMPLATES["factual"]))

        for i in range(num_questions):
            # Build prompt for Q&A generation
            prompt = f"""{template}

Text: {content[:1500]}

Generate a question and answer pair in this format:
Question: [your question]
Answer: [the answer based on the text]"""

            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.8,
                        do_sample=True,
                        num_return_sequences=1
                    )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Parse response
                question, answer = self._parse_qa_response(response, content)

                if question and answer:
                    qa_pairs.append(QAPair(
                        question_id=self._generate_question_id(chunk['chunk_id'], question_type, i),
                        question=question,
                        answer=answer,
                        source_url=chunk.get('url', ''),
                        source_title=chunk.get('title', ''),
                        source_chunk_id=chunk['chunk_id'],
                        question_type=question_type,
                        difficulty=self._estimate_difficulty(question, answer)
                    ))

            except Exception as e:
                print(f"Error generating question: {e}")
                continue

        return qa_pairs

    def _parse_qa_response(self, response: str, context: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse generated QA response."""
        question = None
        answer = None

        response = response.strip()

        # Try to parse Question: and Answer: format
        if "Question:" in response and "Answer:" in response:
            parts = response.split("Answer:")
            q_part = parts[0].replace("Question:", "").strip()
            a_part = parts[1].strip() if len(parts) > 1 else ""

            if q_part and a_part:
                question = q_part
                answer = a_part
        else:
            # Fallback: treat as question and extract answer from context
            if "?" in response:
                question = response.split("?")[0] + "?"
                # Simple answer extraction
                sentences = context.split(".")
                if sentences:
                    answer = sentences[0].strip() + "."

        # Validate
        if question and len(question.split()) > 3 and answer and len(answer) > 5:
            return question, answer

        return None, None

    def generate_dataset(
        self,
        chunks: List[Dict],
        total_questions: int = 100,
        type_distribution: Optional[Dict[str, float]] = None
    ) -> List[QAPair]:
        """
        Generate a diverse Q&A dataset from chunks.

        Args:
            chunks: List of chunk dicts
            total_questions: Total number of questions to generate
            type_distribution: Distribution of question types (e.g., {"factual": 0.4, ...})

        Returns:
            List of QAPair objects
        """
        if type_distribution is None:
            type_distribution = {
                "factual": 0.40,
                "comparative": 0.20,
                "inferential": 0.25,
                "multi-hop": 0.15
            }

        all_qa_pairs = []

        # Calculate questions per type
        questions_per_type = {
            q_type: int(total_questions * fraction)
            for q_type, fraction in type_distribution.items()
        }

        # Shuffle chunks for diversity
        shuffled_chunks = chunks.copy()
        random.shuffle(shuffled_chunks)

        print(f"Generating {total_questions} questions from {len(chunks)} chunks...")

        for q_type, target_count in questions_per_type.items():
            print(f"  Generating {target_count} {q_type} questions...")

            type_qa_pairs = []
            chunk_idx = 0

            with tqdm(total=target_count, desc=f"{q_type}") as pbar:
                while len(type_qa_pairs) < target_count and chunk_idx < len(shuffled_chunks):
                    chunk = shuffled_chunks[chunk_idx]
                    pairs = self.generate_from_chunk(chunk, q_type, 1)
                    type_qa_pairs.extend(pairs)
                    pbar.update(len(pairs))
                    chunk_idx += 1

            all_qa_pairs.extend(type_qa_pairs[:target_count])

        print(f"Generated {len(all_qa_pairs)} Q&A pairs")
        return all_qa_pairs

    def save_dataset(
        self,
        qa_pairs: List[QAPair],
        filename: str = "questions.json"
    ) -> Path:
        """Save Q&A dataset to JSON."""
        output_path = self.eval_dir / filename

        # Calculate statistics
        type_counts = {}
        difficulty_counts = {}

        for pair in qa_pairs:
            type_counts[pair.question_type] = type_counts.get(pair.question_type, 0) + 1
            difficulty_counts[pair.difficulty] = difficulty_counts.get(pair.difficulty, 0) + 1

        data = {
            "total_questions": len(qa_pairs),
            "type_distribution": type_counts,
            "difficulty_distribution": difficulty_counts,
            "questions": [q.to_dict() for q in qa_pairs]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(qa_pairs)} questions to {output_path}")
        return output_path

    def load_dataset(self, filename: str = "questions.json") -> List[QAPair]:
        """Load Q&A dataset from JSON."""
        input_path = self.eval_dir / filename

        with open(input_path, 'r') as f:
            data = json.load(f)

        return [QAPair(**q) for q in data["questions"]]


if __name__ == "__main__":
    # Test question generation
    generator = QuestionGenerator()

    sample_chunks = [
        {
            "chunk_id": "test_001",
            "content": """Python is a high-level, general-purpose programming language.
            Its design philosophy emphasizes code readability with the use of significant indentation.
            Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica
            in the Netherlands as a successor to the ABC programming language.""",
            "url": "https://en.wikipedia.org/wiki/Python",
            "title": "Python (programming language)"
        }
    ]

    pairs = generator.generate_from_chunk(sample_chunks[0], "factual", 2)

    print(f"\nGenerated {len(pairs)} Q&A pairs:")
    for pair in pairs:
        print(f"\nQ: {pair.question}")
        print(f"A: {pair.answer}")
        print(f"Type: {pair.question_type}, Difficulty: {pair.difficulty}")
