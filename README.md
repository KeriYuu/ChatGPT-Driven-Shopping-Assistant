# ChatGPT-Driven-Shopping-Assistant

This project implements a conversational agent designed to assist users with their online shopping needs. The agent takes user inputs, retrieves relevant information or performs actions using various "tools", and provides outputs in a conversational manner. It's also equipped with a memory mechanism to keep track of the conversation history. The agent's behavior is managed by the `Agent` class in `agent.py`, and its interaction with the user is facilitated by the web application implemented in `app.py`.

## Features

- **Smart Retrieval of Tools**: Uses a FAISS-based retriever to select relevant tools based on the user's query.

- **Paraphrase-based Context Understanding**: Uses Language Learning Models (LLMs) to generate paraphrases of the conversation for in-context learning.

- **ReAct Prompting**: LLMs are utilized to generate both reasoning traces and task-specific actions (Thought/ Observation/ Action) in an interleaved manner.

- **Interactive Web Application**: The interactive web application is designed using the Gradio library, providing users with a user-friendly interface to interact with the agent.

## Setup

1. Clone this repository to your local machine.

2. Obtain API keys for Google, OpenAI, RapidAPI, WolframAlpha. Add these keys to a configuration file named `apikey.ini` in the root directory of the project.

3. Install the necessary Python libraries by running `pip install -r requirements.txt` in your terminal.

4. Run `python app.py` to start the web application.

Please note that you need to provide your own API keys for the `Agent` to work, as the keys are required to access various APIs.


## References

S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, "ReAct: Synergizing Reasoning and Acting in Language Models," 
