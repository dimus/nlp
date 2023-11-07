# Importing the SentenceTransformer library
from sentence_transformers import SentenceTransformer


def split_text_into_chunks(text, chunk_size=1000, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# Initializing a SentenceTransformer model with the
# 'multi-qa-mpnet-base-cos-v1' pre-trained model
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

# Defining a list of documents to generate embeddings for
docs = [
    "Around 9 million people live in London",
    "London is known for its financial district"
]

# Generate vector embeddings for the documents

doc_emb = model.encode(
    docs,  # Our documents (an iterable of strings)
    batch_size=32,  # Batch the embeddings by this size
    show_progress_bar=True  # Display a progress bar
)

# The shape of the embeddings is (2, 768), indicating a length of 768 and two
# embeddings generated

print(doc_emb.shape)  # == (2, 768)

with open('../data/seashells_book.txt', 'r') as file:
    txt = file.read()

chunks = split_text_into_chunks(txt)

doc_emb = model.encode(
    chunks,
    batch_size=32,
    show_progress_bar=True
)

print(doc_emb.shape)  # == (2, 768)
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}:\n{chunk}\n\n\n")
