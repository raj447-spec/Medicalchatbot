import truststore
truststore.inject_into_ssl()

import os
os.environ["HF_HUB_OFFLINE"] = "1"

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

existing = [i["name"] for i in pc.list_indexes()]
print(f"Existing indexes ({len(existing)}): {existing}")

if index_name in existing:
    print(f"Index already exists, reusing: {index_name}")
else:
    if len(existing) >= 5:
        to_delete = existing[0]
        print(f"At index quota (5/5). Deleting {to_delete!r} to make room...")
        pc.delete_index(to_delete)
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created index: {index_name}")

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
print("Upsert complete.")
