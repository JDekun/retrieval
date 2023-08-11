from langchain.llms import GPT4All
from langchain.embeddings import GPT4AllEmbeddings

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.vectorstores.faiss import FAISS
from embedding import general_embedding, update_embedding

from langchain import PromptTemplate, LLMChain
import os


# 定义大语言模型LLMs
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
gpt4all_path = './models/ggml-gpt4all-j-v1.3-groovy.bin' 
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)

# 定义嵌入式模型embedding
# embeddings = GPT4AllEmbeddings()

from langchain.embeddings import LlamaCppEmbeddings
llama_path = './models/ggml-model-q4_0.bin' 
embeddings = LlamaCppEmbeddings(model_path=llama_path)

# 生成datastores
base_folder_path = './docs/base'
update_folder_path = './docs/update'
updata = True
if not os.path.exists("datastores"):
    general_embedding(base_folder_path)
elif os.listdir(update_folder_path):
    update_embedding(update_folder_path)


def similarity_search(query, index):
    matched_docs = index.similarity_search(query, k=3) 
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources

# 加载datastores
index = FAISS.load_local("datastores", embeddings)

# 保持连续对话
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

from langchain.chains import ConversationalRetrievalChain
chat = ConversationalRetrievalChain.from_llm(llm, retriever=index.as_retriever(), memory=memory)

while True:
    question = input("Your question: ")
    # question = "What is a PLC and what is the difference with a PC?"
    print(chat.run({"question": question}))
