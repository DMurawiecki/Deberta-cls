import os
import pickle
import sys
from typing import Dict

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings

retriever = None
model = None


def get_retriever_and_model():
    global retriever, model
    if retriever is not None and model is not None:
        return retriever, model

    load_dotenv()
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("MISTRAL_API_KEY not found in environment variables.")
        sys.exit(1)

    current_dir = "./bot"
    books_dir = os.path.join(current_dir, "books")
    EMBEDDINGS_CACHE = os.path.join(books_dir, "embeddings.pkl")
    FAISS_INDEX_FILE = os.path.join(books_dir, "faiss_index")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    if os.path.exists(EMBEDDINGS_CACHE):
        with open(EMBEDDINGS_CACHE, "rb") as f:
            doc_texts, doc_embeddings = pickle.load(f)
    else:
        book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
        documents = []
        for book_file in book_files:
            file_path = os.path.join(books_dir, book_file)
            loader = TextLoader(file_path)
            book_docs = loader.load()
            for doc in book_docs:
                doc.metadata = {"source": book_file}
                documents.append(doc)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        doc_texts = [doc.page_content for doc in docs]
        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
        doc_embeddings = embeddings.embed_documents(doc_texts)
        with open(EMBEDDINGS_CACHE, "wb") as f:
            pickle.dump((doc_texts, doc_embeddings), f)

    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
    else:
        dimension = len(doc_embeddings[0])
        nlist = 100
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(np.array(doc_embeddings, dtype=np.float32))
        index.add(np.array(doc_embeddings, dtype=np.float32))
        faiss.write_index(index, FAISS_INDEX_FILE)

    docs_dict = {
        i: Document(page_content=doc_texts[i], metadata={"source": doc_texts[i][:30]})
        for i in range(len(doc_texts))
    }
    docstore = DictDocstore(docs_dict)
    index_to_docstore_id = {i: str(i) for i in range(len(doc_texts))}
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)

    vector = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    retriever = vector.as_retriever()
    model = ChatMistralAI(mistral_api_key=api_key)
    return retriever, model


class DictDocstore:
    def __init__(self, docs_dict: Dict[int, Document]):
        self.docs_dict: Dict[int, Document] = docs_dict

    def mget(self, keys: list[str]) -> list[Document]:
        return [self.docs_dict[key] for key in keys]

    def mset(self, key_value_pairs: list[tuple[int, Document]]) -> None:
        for key, value in key_value_pairs:
            self.docs_dict[key] = value

    def mdelete(self, keys: list[str]) -> None:
        for key in keys:
            del self.docs_dict[key]

    def yield_keys(self, prefix: str = "") -> list[str]:
        return list(self.docs_dict.keys())

    def search(self, key: str) -> Document:
        return self.docs_dict[int(key)]


class ANSWER:
    def __init__(self, user_input: str):
        retriever, model = get_retriever_and_model()
        prompt_template = """Answer the following question based on the provided context if it is. Do not tell "Based on the provided context" or "The provided context" or something like this:

<context>
{context}
</context>

Question: {input}"""
        prompt = PromptTemplate.from_template(prompt_template)
        document_chain = create_stuff_documents_chain(model, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_input})
        self.answer = response["answer"]

    def __str__(self):
        return self.answer

    def __call__(self):
        return self.answer
