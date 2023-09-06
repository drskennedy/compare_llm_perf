# main.py

import threading
from time import sleep
import os

import LoadVectorize
import LLMPerfMonitor
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import timeit

import warnings
warnings.filterwarnings('ignore')

# child thread to collect stats and add to shared_list every 1 sec
def monitor_thread(event, ppid, shared_list):
    while not event.is_set():
        mem,cpu = LLMPerfMonitor.get_mem_cpu_util(ppid)  # Run the async task in the thread's event loop
        shared_list += [mem,cpu]
        sleep(1)

# parent thread start child, load vectorstore, create a RetrievalQA chain with an LLM and ask queries
def main():
    event = threading.Event()  # Create an event object
    shared_list = []  # Create a shared Queue object
    child = threading.Thread(target=monitor_thread, args=(event,os.getpid(),shared_list))
    child.start()

    db = LoadVectorize.load_db()

    qa_list = LLMPerfMonitor.get_questions_answers()
    qa_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Think step by step before answering.
    Answer:
    """
    prompt = PromptTemplate(template=qa_template, input_variables=['context','question'])
    # console output header
    print('model;question;cosine;exec_time;memory_util;cpu_util')
    # modify list of models directory names to reflect yours  v-----  UPDATE this list
    models = ["alpaca_base","alpaca_gpt4_xl","alpaca_xl","flan_t5_base","flan_t5_xl","flan_sharegpt_xl","LaMini_248M"]
    for model in models:
        model_dir = './models/' + model
        llm = HuggingFacePipeline.from_model_id(model_id=model_dir,
                                                task = 'text2text-generation',pipeline_kwargs={"max_length":450},
                                                model_kwargs={"temperature":0.1,"min_length":35,"repetition_penalty": 4.0})
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}, max_tokens_limit=512), chain_type_kwargs={'prompt': prompt})

        # for each question, ask, compute metrics and empty shared_list
        for i,query in enumerate(qa_list[::2]):
            start = timeit.default_timer()
            result = qa({"query": query})
            time = timeit.default_timer() - start # seconds
            avg_mem = sum(shared_list[::2])/len(shared_list[::2])
            avg_cpu = sum(shared_list[1::2])/len(shared_list[1::2])
            shared_list.clear()
            cos_sim = LLMPerfMonitor.calc_similarity(qa_list[i*2+1],result["result"])
            print(f'{model};{i+1};{cos_sim:.5};{time:.2f};{avg_mem:.2f};{avg_cpu:.2f}')
            #print(f'#{i+1}=>{model};A: {result["result"]}\nSME: {qa_list[i*2+1]};{cos_sim}')

    event.set()  # Set the event to signal the child thread to terminate
    child.join()  # Wait for the child thread to finish

if __name__ == "__main__":
    main()
