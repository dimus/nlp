# Importing the SentenceTransformer library
import os
import logging
from datetime import datetime
import torch

import hashlib
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util
import pinecone

INDEX_NAME = 'bert-search'
# MODEL = 'multi-qa-mpnet-base-cos-v1'
MODEL = 'multi-qa-MiniLM-L6-cos-v1'
pinecone_key = os.environ.get('PINECONE_API_KEY')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

pinecone.init(api_key=pinecone_key, environment="gcp-starter")

if False:
    # INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        INDEX_NAME,  # The name of the index
        dimension=768,  # The dimensionality of the vectors
        metric='cosine',  # The similarity metric to use when
        # searching the index
        pod_type="starter"  # The type of Pinecone pod
    )

# Store the index as a variable
index = pinecone.Index(INDEX_NAME)


def my_hash(s):
    # Return the MD5 hash of the input string as a hexadecimal string
    return hashlib.md5(s.encode()).hexdigest()


def split_text_into_chunks(text, chunk_size=1000, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def prepare_for_pinecone(texts, model=MODEL):
    # Get the current UTC date and time
    now = datetime.utcnow()

    # Generate vector embeddings for each string in the input list, using the
    # specified engine

    embeddings = client.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    # Create tuples of (hash, embedding, metadata) for each input string and
    # its corresponding vector embedding
    # The my_hash() function is used to generate a unique hash for each string,
    # and the datetime.utcnow() function is used to generate the current
    # UTC date and time
    return [
        (
            my_hash(text),  # A unique ID for each string,
            # generated using the my_hash() function
            embedding,  # The vector embedding of the string
            dict(text=text, date_uploaded=now
                 )  # A dictionary of metadata, including the original
            # text and the current UTC date and time
        ) for text, embedding in zip(
            texts, embeddings[0])  # Iterate over each input string and its
        # corresponding vector embedding
    ]


def upload_texts_to_pinecone(texts,
                             batch_size=None,
                             show_progress_bar=False):
    # Call the prepare_for_pinecone function to prepare the
    # input texts for indexing
    total_upserted = 0
    if not batch_size:
        batch_size = len(texts)

    _range = range(0, len(texts), batch_size)
    for i in tqdm(_range) if show_progress_bar else _range:
        batch = texts[i:i + batch_size]
        prepared_texts = prepare_for_pinecone(batch)

        # Use the upsert() method of the index object to upload
        # the prepared texts to Pinecone
        total_upserted += index.upsert(prepared_texts)['upserted_count']

    return total_upserted


def query_from_pinecone(query, top_k=3):
    # get embedding from THE SAME embedder as the documents
    query_embedding = client.encode(
        query,
        batch_size=32,
        show_progress_bar=True
    )

    return index.query(
      vector=query_embedding,
      top_k=top_k,
      include_metadata=True   # gets the metadata (dates, text, etc)
    ).get('matches')


def delete_texts_from_pinecone(texts):
    # Compute the hash (id) for each text
    hashes = [hashlib.md5(text.encode()).hexdigest() for text in texts]

    # The ids parameter is used to specify the list of IDs (hashes) to delete
    return index.delete(ids=hashes)


pinecone.init(api_key=pinecone_key, environment="gcp-starter")

# Initializing a SentenceTransformer model with the
# 'multi-qa-mpnet-base-cos-v1' pre-trained model
client = SentenceTransformer(MODEL)

# The shape of the embeddings is (2, 768), indicating a length of 768 and two
# embeddings generated

with open('../data/christophers_1960.txt', 'r') as file:
    txt = file.read()

texts = split_text_into_chunks(txt)
embeddings = client.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

query = "what viruses Aedes aegypti transmits?"

query_embedded = client.encode(query)

cos_scores = util.cos_sim(query_embedded, embeddings)[0]
top_k = min(5, len(texts))
top_results = torch.topk(cos_scores, k=top_k)

print("\n\n======================\n\n")
print("Query:", query)
print("\nTop 5 most similar sentences in corpus:")

for score, idx in zip(top_results[0], top_results[1]):
    if score < -0.6:
        continue
    print(texts[idx], "(Score: {:.4f})".format(score))
    print("\n\n======================\n\n")

# res = prepare_for_pinecone(texts)
# uploaded = upload_texts_to_pinecone(texts)
# res = query_from_pinecone("shells used to make purpur color")
# print(res) # == (2, 768)
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}:\n{chunk}\n\n\n")
