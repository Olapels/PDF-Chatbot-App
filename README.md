# PDF ChatBot App

### About the Project
This is a Chatbot App that allows users to upload PDF files and interact with the contents conversationally. Under the hood, the app is built on top of existing frameworks of LLMs(Large Language Models). 



The app was built entirely in
* [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)

Using the following libraries

* [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
* [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
* [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFAC2D?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)
* #### [🦜️🔗 LangChain](https://www.langchain.com)

### System Requirements
Due to the large size and memory requirements for running llms, it is hugely recommended to run this first in the cloud using services such as:
* Google Colab
* GitHub Codespaces

To run the app locally, the following should be the minimum requirement
### Setup
clone the repo into your desired folder

Using HTTPS
```
git clone https://github.com/Olapels/PDF-Chatbot-App.git
```
Using CLI
```
gh repo clone Olapels/PDF-Chatbot-App
```
Install the required dependencies contained in requirements.txt
```
pip install -r requirements.txt
```
change current working directory to the models folder by running
```
cd models
```
 to  download the underlying llm (.gguf quantized llama model) by [theBloke](https://huggingface.co/TheBloke) run the code below
```
wget https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/blob/main/codellama-7b.Q2_K.gguf
```

The underlying model used is the .gguf type quantization of the open source llama model by Meta which is free for commercial use up to a certain limit.

### Running the App
in your terminal/command prompt (on the main project folder) run 
```
streamlit run app.py
```


