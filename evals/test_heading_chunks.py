import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chunker.chunking import process_file


INPUT_DIR = PROJECT_ROOT / "data" / "structured_final"

total_chunks = 0
heading_only_chunks = []


for file_path in sorted(INPUT_DIR.glob("*.txt")):

    chunks = process_file(file_path)

    for i, chunk in enumerate(chunks, start=1):

        total_chunks += 1

        content = chunk["content"].strip()

        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        # Remove markdown heading lines
        non_heading_lines = [
            line
            for line in lines
            if not line.startswith("#")
        ]

        if not non_heading_lines:

            heading_only_chunks.append({
                "file": file_path.name,
                "chunk_number": i,
                "content": content
            })


print("\n" + "=" * 60)
print("HEADING-ONLY CHUNK TEST")
print("=" * 60)

print(f"Total chunks: {total_chunks}")
print(f"Heading-only chunks: {len(heading_only_chunks)}")


if heading_only_chunks:

    print("\nFOUND:\n")

    for item in heading_only_chunks:

        print("-" * 60)
        print(f"File: {item['file']}")
        print(f"Chunk: {item['chunk_number']}")
        print(item["content"])

else:

    print("\nNo heading-only chunks found.")


print("=" * 60)