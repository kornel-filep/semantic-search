import warnings

warnings.filterwarnings("ignore")

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

import os
import time
import torch

from tqdm.auto import tqdm

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY')

dataset = load_dataset('quora', split='train[240000:290000]')

# Get only questions from dataset
questions = []
for record in dataset['questions']:
    questions.extend(record['text'])
question = list(set(questions))
print(f'Number of questions: {len(questions)}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print('Sorry no cuda.')

# Use a lightweight model for faster processing
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = f'semantic-search-experimentation'

# delete index if it exists, then create a new one in pinecone
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)
print(INDEX_NAME)
pinecone.create_index(name=INDEX_NAME, 
    dimension=model.get_sentence_embedding_dimension(), 
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pinecone.Index(INDEX_NAME)
print(index)

batch_size=200
vector_limit=10000

questions = question[:vector_limit]

import json

# upsert records in batches of 200 to pinecone index
for i in tqdm(range(0, len(questions), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(questions))
    # create IDs batch
    ids = [str(x) for x in range(i, i_end)]
    # create metadata batch
    metadatas = [{'text': text} for text in questions[i:i_end]]
    # create embeddings
    embeddings = model.encode(questions[i:i_end])
    # create records list for upsert
    records = zip(ids, embeddings, metadatas) # zip converts everything to a tuple with (id, embedding, metadata) format
    # upsert to Pinecone
    index.upsert(vectors=records)

print(index.describe_index_stats())

def run_query(query):
  embedding = model.encode(query).tolist()
  results = index.query(top_k=10, vector=embedding, include_metadata=True, include_values=False)
  for result in results['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")

run_query('which city has the highest population in the world?')

