import tools as search
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from utils import read_funcs

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class Retriever:
    def __init__(self, debug=False, max_tools=10):
        self.All_tools = []
        self.max_tools = max_tools
        self.llm = ChatOpenAI(temperature=0)
        self.debug = debug
        search_apis = read_funcs("src/tools.py")
        for name in search_apis:
            self.All_tools.append(getattr(tools, name))

        docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(self.All_tools)]
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        self.retriever = vector_store.as_retriever(k=self.max_tools)

    def get_tools(self, query):
        messages = []
        temp = ""
        # load the examples for in-context-learning
        with open("src/prompts/paraphrase.txt") as f:
            line = f.readline()
            turn = 0
            while line:
                if line.startswith("system:"):
                    turn += 1
                    temp = line[7:]
                elif line.startswith("input:"):
                    turn += 1
                    if turn == 2:
                        messages.append(SystemMessage(content=temp))
                    else:
                        messages.append(AIMessage(content=temp))
                    temp = line[6:]
                elif line.startswith("output:"):
                    turn += 1
                    messages.append(HumanMessage(content=temp))
                    temp = line[7:]
                else:
                    temp += line
                line = f.readline()
            if turn == 1:
                messages.append(SystemMessage(content=temp))
            else:
                messages.append(AIMessage(content=temp))

        messages.append(HumanMessage(content=query))
        paraphrase = self.llm(messages).content
        if self.debug:
            print("paraphrase is:", paraphrase)
        docs = self.retriever.get_relevant_documents(query + "\n" + paraphrase)
        tools = [self.All_tools[d.metadata["index"]] for d in docs]
        if tools.human_feedback not in tools:
            tools = tools + [tools.human_feedback]

        if self.debug:
            print("proposal tools:")
            for tool in tools:
                print("tool_name:", tool.name)
        return tools
