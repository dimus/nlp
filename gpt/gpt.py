import openai
# from openai.embeddings_utils import get_embeddings, get_embedding
from datetime import datetime
import hashlib
import re
import os
from tqdm import tqdm
import numpy as np
import pinecone

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

openai.api_key = os.environ.get('OPENAI_API_KEY')
pinecone_key = os.environ.get('PINECONE_API_KEY')

INDEX_NAME = 'semantic-search'
NAMESPACE = 'default'
ENGINE = 'text-embedding-ada-002'

pinecone.init(api_key=pinecone_key, environment="gcp-starter")

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        INDEX_NAME,  # The name of the index
        dimension=1536,  # The dimensionality of the vectors
        metric='cosine',  # The similarity metric to use when
        # searching the index
        pod_type="p1"  # The type of Pinecone pod
    )

# Store the index as a variable
index = pinecone.Index(INDEX_NAME)

client = openai.OpenAI()


def get_embedding(text, model=ENGINE):
    text = text.replace("\n", " ")
    res = client.embeddings.create(input=[text], model=model)
    return res.data[0].embedding


def get_embeddings(texts, model=ENGINE):
    res = []
    for i, text in enumerate(texts):
        res.append(get_embedding(text, model))

    return res


def my_hash(s):
    # Return the MD5 hash of the input string as a hexadecimal string
    return hashlib.md5(s.encode()).hexdigest()


def prepare_for_pinecone(texts, model=ENGINE):
    # Get the current UTC date and time
    now = datetime.utcnow()

    # Generate vector embeddings for each string in the input list, using the
    # specified engine

    embeddings = get_embeddings(texts, model=ENGINE)

    # Create tuples of (hash, embedding, metadata) for each input string and
    # its corresponding vector embedding
    # The my_hash() function is used to generate a unique hash for each string,
    # and the datetime.utcnow() function is used to generate the current
    # UTC date and time
    return [
        (
            my_hash(
                text
            ),  # A unique ID for each string, 
                # generated using the my_hash() function
            embedding,  # The vector embedding of the string
            dict(
                text=text, date_uploaded=now
            )  # A dictionary of metadata, including the original
               # text and the current UTC date and time
        ) for text, embedding in zip(
            texts, embeddings
        )  # Iterate over each input string and its
           # corresponding vector embedding
    ]


texts = ["hi"]
res = prepare_for_pinecone(texts)
print(res)
