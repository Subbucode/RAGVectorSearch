from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders  import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import time



model = SentenceTransformer('nomic-ai/nomic-embed-text-v1',trust_remote_code=True)


# Define a function to generate embeddings
def generate_embeddings(data):
    """Generates vector embeddings for the given data."""
    embeddings = model.encode(data)

    return embeddings.tolist()

#Load the pdf
loader = PyPDFLoader('https://investors.mongodb.com/node/12236/pdf')
data = loader.load()
print("data loaded")
time.sleep(5)

#spilt the data into chucks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
documents = text_splitter.split_documents(data)

# Prepare documents for insertion
# Prepare documents for insertion
docs_to_insert = [{
    "text": doc.page_content,
    "embedding": generate_embeddings(doc.page_content)
} for doc in documents]
print("documents prepared")
time.sleep(5)
#connect to altas cluster

client = MongoClient("mongodb+srv://subbutraining:Mymongouser@cluster0.jdrr2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
collection = client["rag_db"]["test"]
print("connected to cluster")
time.sleep(5)
# Insert documents into the collection
result = collection.insert_many(docs_to_insert)

print(f"Inserted {len(result.inserted_ids)} documents.")

print("creating the index model")

# Create your index model, then create the search index
index_name="vector_index"
search_index_model = SearchIndexModel(
  definition = {
    "fields": [
      {
        "type": "vector",
        "numDimensions": 768,
        "path": "embedding",
        "similarity": "cosine"
      }
    ]
  },  name = index_name,
  type = "vectorSearch"
)
collection.create_search_index(search_index_model)
print("Index created")
time.sleep(5)
# Perform a search
# Wait for initial sync to complete
print("Polling to check if the index is ready. This may take up to a minute.")
predicate=None
if predicate is None:
   predicate = lambda index: index.get("queryable") is True

while True:
   indices = list(collection.list_search_indexes(index_name))
   if len(indices) and predicate(indices[0]):
      break
   time.sleep(5)
print(index_name + " is ready for querying.")   
   time.sleep(5)
# Define a function to run vector search queries
# Define a function to run vector search queries
def get_query_results(query):
  """Gets results from a vector search query."""

  query_embedding = generate_embeddings(query)
  pipeline = [
      {
            "$vectorSearch": {
              "index": "vector_index",
              "queryVector": query_embedding,
              "path": "embedding",
              "exact": True,
              "limit": 5
            }
      }, {
            "$project": {
              "_id": 0,
              "text": 1
         }
      }
  ]

  results = collection.aggregate(pipeline)

  array_of_results = []
  for doc in results:
      array_of_results.append(doc)
  return array_of_results

# Test the function with a sample query
import pprint
pprint.pprint(get_query_results("AI technology"))