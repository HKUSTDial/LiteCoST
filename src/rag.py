'''Use RAG model'''
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma, FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
import os, dotenv, random, json
from tqdm import tqdm
import torch


os.environ["OPENAI_API_BASE"] = ''
os.environ["OPENAI_API_KEY"] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RAG:
    def __init__(self, question: str, docs: str, model: str, top_k: int):
        prompt = ChatPromptTemplate.from_template("""
        You are a Q&A chatbot assistant. Please use the retrieved context below to answer the question. If you do not know the answer, just say you do not know.
        <context>
        {context}
        </context>
        ---
        Question: {input}                                                                                                                 
        """)
        llm = ChatOpenAI(model_name=model, temperature=0)
        self.question = question
        self.docs = [Document(page_content=docs)]
        self.top_k = top_k
        self.document_chain = create_stuff_documents_chain(llm, prompt)
        self.retriever = self.get_retriever()

    def get_retriever(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        chunks = text_splitter.split_documents(self.docs)
        # print(len(chunks),type(chunks))

        model_name = "BAAI/bge-m3"
        model_kwargs = {'device': 'cuda:4'}
        embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder= "/data/liangzhuowen/hf_models",
                # model_kwargs=model_kwargs
        )
        # embedding=OpenAIEmbeddings(model="text-embedding-3-large")
        vectorstore = FAISS.from_documents(chunks,embedding_model)

        retriever = vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        return retriever

    def get_chunks(self):
        results = self.retriever.invoke(self.question)
        # print(len(results))
        return [result.page_content for result in results]

    def get_response(self):
        # rag_chain = (
        #     {"context": retriever, "question": RunnablePassthrough()}
        #     | prompt
        #     | llm
        #     | StrOutputParser()
        # )
        # # 查询问题
        # query = "萧炎表妹是谁？"
        # res = rag_chain.invoke(query)
        
        print(self.retriever)
        retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)
        result = retrieval_chain.invoke({"input": self.question})
        # print(f'get_response = {question}, {result}')
        return result['answer']
