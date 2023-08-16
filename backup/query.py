# -*- coding: utf-8 -*-

from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores.faiss import FAISS
from embedding import general_embedding, update_embedding
import os

gpt4all_path = '../models/ggml-gpt4all-j-v1.3-groovy.bin' 
# gpt4all_path = './models/wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin' 
llama_path = '../models/ggml-model-q4_0.bin' 


# 定义嵌入式模型embedding
embeddings = LlamaCppEmbeddings(model_path=llama_path)

# 定义大语言模型LLMs
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)

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

# 生成datastores
base_folder_path = './docs/base'
update_folder_path = './docs/update'
updata = True
if not os.path.exists("datastores"):
    general_embedding(base_folder_path)
elif os.listdir(update_folder_path):
    update_embedding(update_folder_path)


# 加载datastores
index = FAISS.load_local("datastores", embeddings)

# 保持连续对话
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# prompt 样例
template = """Please use the following context to answer questions.
Context: {context}
------------------------
Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 定义链
llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)

while True:
    # query
    question = input("Your question: ")

    # 检索，得到上下文信息context
    matched_docs, sources = similarity_search(question, index)
    context = "\n".join([doc.page_content for doc in matched_docs])

    llm_chain.prompt = llm_chain.prompt.partial(context=context)

    # 将检索结果和问题一起输入LLM，输出结果
    print(llm_chain.run(question))