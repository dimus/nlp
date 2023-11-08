# Importing the SentenceTransformer library
import os
import logging
import torch
import psycopg2

import hashlib

from sentence_transformers import SentenceTransformer, util
import pinecone

INDEX_NAME = 'bert-search'
# MODEL = 'multi-qa-mpnet-base-cos-v1'
MODEL = 'multi-qa-MiniLM-L6-cos-v1'
pinecone_key = os.environ.get('PINECONE_API_KEY')
db_name = os.environ.get("NLP_DB_NAME")
db_host = os.environ.get("NLP_DB_HOST")
db_user = os.environ.get("NLP_DB_USER")
db_pass = os.environ.get("NLP_DB_PASS")
db_port = os.environ.get("NLP_DB_PORT")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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


# Initializing a SentenceTransformer model with the
# 'multi-qa-mpnet-base-cos-v1' pre-trained model
client = SentenceTransformer(MODEL)

# The shape of the embeddings is (2, 768), indicating a length of 768 and two
# embeddings generated

with open('../data/seashells_book.txt', 'r') as file:
    txt = file.read()

texts = split_text_into_chunks(txt)
embeddings = client.encode(texts, batch_size=32, show_progress_bar=True)

db = psycopg2.connect(database=db_name,
                      host=db_host,
                      user=db_user,
                      password=db_pass,
                      port=db_port)

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
