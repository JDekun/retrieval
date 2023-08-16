from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# function for loading only TXT files
from langchain.document_loaders import TextLoader
# text splitter for create chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# to be able to load the pdf files
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
# Vector Store Index to create our database about our knowledge
from langchain.indexes import VectorstoreIndexCreator
# LLamaCpp embeddings from the Alpaca model
from langchain.embeddings import LlamaCppEmbeddings
# FAISS  library for similaarity search
from langchain.vectorstores.faiss import FAISS
import datetime
import os, time, shutil


def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def create_index(chunks, embeddings):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return search_index


def general_embedding(base_folder_path, embedding_path):
    # 定义嵌入式模型embedding

    embeddings = LlamaCppEmbeddings(model_path=embedding_path)

    doc_list = [s for s in os.listdir(base_folder_path) if s.endswith('.pdf')]
    num_of_docs = len(doc_list)

    # create a loader for the PDFs from the path
    general_start = datetime.datetime.now() 
    print("starting the loop...")
    loop_start = datetime.datetime.now() 
    print("generating fist vector database and then iterate with .merge_from")
    loader = PyPDFLoader(os.path.join(base_folder_path, doc_list[0]))
    docs = loader.load()
    chunks = split_chunks(docs)
    db0 = create_index(chunks, embeddings)
    print("Main Vector database created. Start iteration and merging...")
    for i in range(1,num_of_docs):
        print(doc_list[i])
        print(f"loop position {i}/{num_of_docs}")
        loader = PyPDFLoader(os.path.join(base_folder_path, doc_list[i]))
        start = datetime.datetime.now() 
        docs = loader.load()
        chunks = split_chunks(docs)
        dbi = create_index(chunks, embeddings)
        print("start merging with db0...")
        db0.merge_from(dbi)
        end = datetime.datetime.now() 
        elapsed = end - start 
        #total time
        print(f"completed in {elapsed}")
        print("-----------------------------------")
    loop_end = datetime.datetime.now() 
    loop_elapsed = loop_end - loop_start 
    print(f"All documents processed in {loop_elapsed}")
    print(f"the daatabase is done with {num_of_docs} subset of db index")
    print("-----------------------------------")
    print(f"Merging completed")
    print("-----------------------------------")
    print("Saving Merged Database Locally")

    # 保存datastores
    db0.save_local("datastores")

    print("-----------------------------------")
    print("merged database saved as datastores")
    general_end = datetime.datetime.now() 
    general_elapsed = general_end - general_start 
    print(f"All indexing completed in {general_elapsed}")
    print("-----------------------------------")
    
def update_embedding(update_folder_path, embedding_path):
    # 定义嵌入式模型embedding
 
    embeddings = LlamaCppEmbeddings(model_path=embedding_path)

    doc_list = [s for s in os.listdir(update_folder_path) if s.endswith('.pdf')]
    num_of_docs = len(doc_list)

    # 加载datastores
    db0 = FAISS.load_local("datastores", embeddings)

    # create a loader for the PDFs from the path
    start = datetime.datetime.now() 

    print("Base Vector database loaded. Start iteration and updating...")
    for i in range(0,num_of_docs):
        print(doc_list[i])
        print(f"loop position {i}/{num_of_docs}")
        loader = PyPDFLoader(os.path.join(update_folder_path, doc_list[i]))
        start = datetime.datetime.now() 
        docs = loader.load()
        chunks = split_chunks(docs)
        dbi = create_index(chunks, embeddings)
        print("start merging with db0...")
        db0.merge_from(dbi)
        end = datetime.datetime.now() 
        elapsed = end - start 
        #total time
        print(f"completed in {elapsed}")
        print("-----------------------------------")
    loop_end = datetime.datetime.now() 
    loop_elapsed = loop_end - start 
    print(f"All documents processed in {loop_elapsed}")
    print(f"the daatabase is done with {num_of_docs} subset of db index")
    print("-----------------------------------")
    print(f"Merging completed")
    print("-----------------------------------")
    print("Saving Merged Database Locally")

    # 保存datastores
    db0.save_local("datastores")

    print("-----------------------------------")
    print("merged database saved as datastores")
    general_end = datetime.datetime.now() 
    general_elapsed = general_end - start 
    print(f"All indexing completed in {general_elapsed}")
    print("-----------------------------------")


    dirname = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    update_doc_save_path = 'docs/' + 'update_' + dirname
    os.mkdir(update_doc_save_path)

    filelist = os.listdir(update_folder_path) 
    for file in filelist:
        old = os.path.join(update_folder_path, file)
        new = os.path.join(update_doc_save_path, file)
        shutil.move(old, new)


