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

    while True:

        question = input("Question: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        chunks = retrieve_chunks(question,
                                 max_candidates=MAX_CANDIDATES,
                                 distance_threshold=DISTANCE_THRESHOLD)

        answer = generate_answer(question, chunks)

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
