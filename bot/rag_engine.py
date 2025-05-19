import os
import pickle
import stat
import sys
from typing import Dict
import shutil
import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv


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
        load_dotenv()
        api_key = os.environ.get("MISTRAL_API_KEY")

        if not api_key:
            print("MISTRAL_API_KEY not found in environment variables.")
            sys.exit(1)

        current_dir = "/Users/tadeuskostusko/Documents/Deberta-cls/bot"
        books_dir = os.path.join(current_dir, "books")
        db_dir = os.path.join(current_dir, "db")
        persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

        if not os.path.exists(persistent_directory):
            if not os.path.exists(books_dir):
                raise FileNotFoundError(
                    f"The directory {books_dir} does not exist. Please check the path."
                )

            book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
            documents = []
            for book_file in book_files:
                file_path = os.path.join(books_dir, book_file)
                loader = TextLoader(file_path)
                book_docs = loader.load()
                for doc in book_docs:
                    doc.metadata = {"source": book_file}
                    documents.append(doc)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0
            )
            docs = text_splitter.split_documents(documents)
        else:
            pass

        EMBEDDINGS_CACHE = "./embeddings.pkl"
        FAISS_INDEX_FILE = "./faiss_index"

        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)

        if os.path.exists(EMBEDDINGS_CACHE) and os.path.getsize(EMBEDDINGS_CACHE) == 0:
            os.remove(EMBEDDINGS_CACHE)

        if os.path.exists(EMBEDDINGS_CACHE):
            with open(EMBEDDINGS_CACHE, "rb") as f:
                doc_texts, doc_embeddings = pickle.load(f)
        else:
            doc_texts = [doc.page_content for doc in docs]
            doc_embeddings = embeddings.embed_documents(doc_texts)
            with open(EMBEDDINGS_CACHE, "wb") as f:
                pickle.dump((doc_texts, doc_embeddings), f)

        if os.path.exists(FAISS_INDEX_FILE):
            try:
                if os.path.isdir(FAISS_INDEX_FILE):
                    shutil.rmtree(FAISS_INDEX_FILE)
                else:
                    os.remove(FAISS_INDEX_FILE)
            except PermissionError:
                os.chmod(FAISS_INDEX_FILE, stat.S_IWUSR)
                if os.path.isdir(FAISS_INDEX_FILE):
                    shutil.rmtree(FAISS_INDEX_FILE)
                else:
                    os.remove(FAISS_INDEX_FILE)

        dimension = len(doc_embeddings[0])
        nlist = 100
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        index.train(np.array(doc_embeddings, dtype=np.float32))
        index.add(np.array(doc_embeddings, dtype=np.float32))

        docs_dict = {
            i: Document(
                page_content=doc_texts[i], metadata={"source": doc_texts[i][:30]}
            )
            for i in range(len(doc_texts))
        }
        docstore: DictDocstore = DictDocstore(docs_dict)
        index_to_docstore_id = {i: str(i) for i in range(len(doc_texts))}

        vector = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        vector.save_local(FAISS_INDEX_FILE)
        retriever = vector.as_retriever()
        model = ChatMistralAI(mistral_api_key=api_key)

        prompt_template = """Answer the following question based only on the provided context. Do not tell "Based on the provided context" or "The provided context" or
        something like this:

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
