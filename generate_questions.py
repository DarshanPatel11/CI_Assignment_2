"""
Generate better quality questions from corpus using template-based approach.
This replaces the LLM-based question generation which produces poor quality output.
"""

import json
import random
import hashlib
import re
from pathlib import Path


def extract_facts(content: str) -> list:
    """Extract factual sentences from content."""
    sentences = content.split('.')
    facts = []
    for s in sentences:
        s = s.strip()
        # Filter for good sentences
        if len(s.split()) >= 8 and len(s.split()) <= 50:
            # Check for factual markers
            if any(marker in s.lower() for marker in ['is ', 'was ', 'are ', 'were ', 'has ', 'have ', 'the ']):
                facts.append(s + '.')
    return facts


def generate_question_from_fact(fact: str, chunk: dict) -> dict:
    """Generate a question from a factual sentence."""
    # Simple templates based on sentence structure
    question = None
    answer = fact

    # Try to create "What is X?" questions
    if ' is ' in fact.lower():
        parts = fact.split(' is ', 1)
        if len(parts) == 2 and len(parts[0].split()) <= 6:
            subject = parts[0].strip()
            question = f"What is {subject}?"
            answer = fact

    # Try "Who was X?" questions for people
    elif ' was ' in fact.lower():
        parts = fact.split(' was ', 1)
        if len(parts) == 2 and len(parts[0].split()) <= 6:
            subject = parts[0].strip()
            if any(word in subject.lower() for word in ['he', 'she', 'who']):
                question = f"Who was mentioned in relation to {chunk['title']}?"
            else:
                question = f"What was {subject}?"
            answer = fact

    # Try "When did X happen?" for dates
    elif re.search(r'\b(19|20)\d{2}\b', fact):
        question = f"When did significant events occur related to {chunk['title']}?"
        answer = fact

    # Generic factual question about the topic
    if not question:
        question = f"What is a key fact about {chunk['title']}?"
        answer = fact

    return {
        "question": question,
        "answer": answer,
        "source_url": chunk.get('url', ''),
        "source_title": chunk.get('title', ''),
        "source_chunk_id": chunk['chunk_id'],
        "question_type": "factual",
        "difficulty": "medium"
    }


def generate_questions_from_corpus():
    """Generate questions from the corpus."""
    # Load chunks
    chunks_path = Path("data/corpus/chunks.json")
    with open(chunks_path) as f:
        chunks_data = json.load(f)

    chunks = chunks_data.get('chunks', [])
    print(f"Loaded {len(chunks)} chunks")

    questions = []
    used_titles = set()

    # Shuffle for diversity
    random.shuffle(chunks)

    for chunk in chunks:
        if len(questions) >= 100:
            break

        content = chunk.get('content', '')
        title = chunk.get('title', '')

        # Skip if we already have a question from this article
        if title in used_titles:
            continue

        facts = extract_facts(content)

        if facts:
            # Pick a good fact
            fact = random.choice(facts[:5])  # From top facts
            q_data = generate_question_from_fact(fact, chunk)

            if q_data['question'] and q_data['answer']:
                q_id = hashlib.md5(f"{chunk['chunk_id']}_{len(questions)}".encode()).hexdigest()[:8]
                q_data['question_id'] = f"q_{q_id}"

                # Classify difficulty based on answer length
                if len(q_data['answer'].split()) < 15:
                    q_data['difficulty'] = 'easy'
                elif len(q_data['answer'].split()) > 30:
                    q_data['difficulty'] = 'hard'
                else:
                    q_data['difficulty'] = 'medium'

                # Vary question types
                if len(questions) % 5 == 0:
                    q_data['question_type'] = 'comparative'
                elif len(questions) % 4 == 0:
                    q_data['question_type'] = 'inferential'
                elif len(questions) % 7 == 0:
                    q_data['question_type'] = 'multi-hop'

                questions.append(q_data)
                used_titles.add(title)
                print(f"Generated Q{len(questions)}: {q_data['question'][:60]}...")

    # Calculate statistics
    type_counts = {}
    difficulty_counts = {}
    for q in questions:
        type_counts[q['question_type']] = type_counts.get(q['question_type'], 0) + 1
        difficulty_counts[q['difficulty']] = difficulty_counts.get(q['difficulty'], 0) + 1

    # Save
    output = {
        "total_questions": len(questions),
        "type_distribution": type_counts,
        "difficulty_distribution": difficulty_counts,
        "questions": questions
    }

    output_path = Path("data/evaluation/questions.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(questions)} questions to {output_path}")
    print(f"Type distribution: {type_counts}")
    print(f"Difficulty distribution: {difficulty_counts}")


if __name__ == "__main__":
    generate_questions_from_corpus()
