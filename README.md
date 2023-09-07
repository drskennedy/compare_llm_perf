#Multithreaded Test Framework for Comparing Local LLM Performance for Document Q&A

**Step-by-step guide on Medium**: [Comparing LLM Performance](https://medium.com/@heelara/multi-threaded-framework-for-testing-language-models-for-question-answering-664fdd31b111)
___
## Context
Even through there are third-party commercial large language model (LLM) providers like OpenAI's GPT4 have made it easy to access LLM via simple API calls, privacy-conscious researchers and engineers are looking to deploy a fully local model to allow querying against their local documents to main their intellectual properties with their walls. The proliferation of open-source LLMs is providing us with many options, which may be daunting.
In this project, we will introduce a multithreaded test framework and the associated code to more objectively compare the models to find the best LLM for our constraints.
<br><br>
![Thread Diagram](/assets/thread_diagram.png)
___
## How to Install
- Create and activate the environment:
```
$ python3.10 -m venv mychat
$ source mychat/bin/activate
```
- Install libraries:
```
$ pip install -r requirements.txt
```
- Download one or more models (`text2textgeneration`) to directory `models` manually or using script `download_model.py`. For instance, to download files of 'declare-lab/flan-alpaca-base', it can be launched like this:
```
$ python download_model.py 'declare-lab/flan-alpaca-base' alpaca_base
```
- Ensure the data structure list models in `main.py` reflects the models that you have downloaded.
- Run script `main.py` to start the testing:
```
$ python main.py
```
___
## Quickstart
- To start the app, launch terminal from the project directory and run the following command:
```
$ source mychat/bin/activate
$ python main.py
```
- Here is a sample run:
```
$ python main.py
model;question;cosine;exec_time;memory_util;cpu_util
alpaca_base;1;0.50597;1.61;1.18;151.68
alpaca_base;2;0.65736;1.47;1.86;133.95
alpaca_base;3;0.78416;2.57;1.86;134.37
alpaca_base;4;0.66734;1.17;1.85;134.40
alpaca_base;5;0.74257;2.07;1.85;121.70
alpaca_base;6;0.66274;2.39;1.91;121.23
alpaca_base;7;0.21965;1.02;1.92;140.70
alpaca_base;8;0.57093;1.99;1.94;132.15
alpaca_base;9;0.49714;1.52;1.94;121.80
alpaca_base;10;0.59772;1.69;1.94;125.30
...
```
___
## Tools
- **LangChain**: Framework for developing applications powered by language models
- **FAISS**: Open-source library for efficient similarity search and clustering of dense vectors.
- **Sentence-Transformers (all-MiniLM-L6-v2)**: Open-source pre-trained transformer model for embedding text to a dense vector space for tasks like cosine similarity calculation.

___
## Files and Content
- `models`: Directory hosting sub-directories of downloaded LLMs
- `opdf_index`: FAISS vectorstore for documents
- `main.py`: Main Python script to launch the application and to pass user query via command line
- `LoadVectorize.py`: Python script to load a pdf document, split and vectorize
- `LLMPerfMonitor.py`: Python script to calculate metrics and load a list of question
- `download_model.py`: Python script to download an LLM model from HuggingFace, requires environment variable "HF_API_KEY" to be set
- `requirements.txt`: List of Python dependencies (and version)
___

## References
- https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
