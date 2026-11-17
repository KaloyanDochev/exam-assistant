from collections import deque
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

from rich.console import Console
from rich.markdown import Markdown

from retrieval import retrieve_chunks
from generation import generate_answer

load_dotenv()

SAVE_ANSWERS = os.getenv("SAVE_ANSWERS", "true").lower() == "true"

ANSWER_DIRECTORY = Path(os.getenv("ANSWER_DIRECTORY", "answers"))

MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "20"))

DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "1.5"))

ENABLE_CONVERSATION_MEMORY = (os.getenv("ENABLE_CONVERSATION_MEMORY",
                                        "true").lower() == "true")

MAX_MEMORY_EXCHANGES = int(os.getenv("MAX_MEMORY_EXCHANGES", "5"))


def save_answer(question: str, answer: str) -> Path:
    """Save a question and its answer to a Markdown file."""

    output_directory = ANSWER_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_directory / f"answer_{timestamp}.md"

    content = f"""# Question

{question}

# Answer

{answer}
"""

    file_path.write_text(content, encoding="utf-8")

    return file_path


def main():

    print("================================")
    print("       EXAM STUDY ASSISTANT")
    print("================================")
    print("Enter 'exit' to quit.\n")

    conversation_memory = (deque(
        maxlen=MAX_MEMORY_EXCHANGES) if ENABLE_CONVERSATION_MEMORY else None)

    while True:

        question = input("Question: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        retrieved_chunks = retrieve_chunks(
            question,
            max_candidates=MAX_CANDIDATES,
            distance_threshold=DISTANCE_THRESHOLD,
            conversation_memory=conversation_memory)

        answer = generate_answer(question,
                                 retrieved_chunks,
                                 conversation_memory=conversation_memory)

        if conversation_memory is not None and answer is not None:
            conversation_memory.append({
                "question": question,
                "answer": answer
            })

        print("\nAnswer:\n")

        if answer is None:
            print(
                "I couldn't find relevant information in the provided study materials."
            )
        else:
            console = Console()
            console.print(Markdown(answer))
            if SAVE_ANSWERS:
                file_path = save_answer(question, answer)
                print(f"\nAnswer saved to: {file_path}")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
