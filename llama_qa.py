from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.vectorstores.faiss import FAISS
from embedding import general_embedding, update_embedding

from langchain import PromptTemplate, LLMChain
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from langchain import HuggingFacePipeline
import torch


# 定义大语言模型LLMs
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# model_path="./models/llama-2-7b-chat.ggmlv3.q8_0.bin"
# llm = LlamaCpp(
#     model_path=model_path,
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     n_ctx=2048,
#     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#     callback_manager=callback_manager,
#     verbose=True,
# )

model_path ="./models/chinese-llama-2-7b"
llm = HuggingFacePipeline.from_model_id(model_id=model_path,
        task="text-generation",
        device=0,
        model_kwargs={
                        "torch_dtype" : torch.float16,
                        "low_cpu_mem_usage" : True,
                        "temperature": 0.2,
                        "max_length": 1000,
                        "repetition_penalty":1.1}
        )

# 定义嵌入式模型embeddin# embedding
embedding_path = "./models/llama-2-7b-chat.ggmlv3.q8_0.bin"
embeddings = LlamaCppEmbeddings(model_path=embedding_path)

# 生成datastores
base_folder_path = './docs/base'
update_folder_path = './docs/update'
updata = True
if not os.path.exists("datastores"):
    general_embedding(base_folder_path, embedding_path)
elif os.listdir(update_folder_path):
    update_embedding(update_folder_path, embedding_path)


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
    print(chat.run({"question": question}))
