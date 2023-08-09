import tools as search_tools
from utils import read_funcs
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage, Document
from langchain.chat_models import ChatOpenAI


class Retriever:
    def __init__(self, debug_mode=False, tool_limit=10):
        self.All_tools = []
        self.tool_limit = tool_limit
        self.llm = ChatOpenAI(temperature=0)
        self.debug_mode = debug_mode
        
        # Reading available tools from the tools module
        available_tools = read_funcs("src/tools.py")
        for tool_name in available_tools:
            self.All_tools.append(getattr(search_tools, tool_name))

        docs = [Document(page_content=tool.description, metadata={"index": index}) for index, tool in enumerate(self.All_tools)]
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        self.retriever_engine = vector_store.as_retriever(k=self.tool_limit)

    def retrieve_tools(self, query):
        # Load example prompts for in-context-learning
        messages = self._load_prompts("src/prompts/paraphrase.txt")
        
        # Add the given query to the prompts
        messages.append(HumanMessage(content=query))
        
        # Get paraphrased query using the llm model
        paraphrased_query = self.llm(messages).content
        
        if self.debug_mode:
            print("Paraphrased query:", paraphrased_query)
        
        relevant_documents = self.retriever_engine.get_relevant_documents(query + "\\n" + paraphrased_query)
        relevant_tools = [self.All_tools[doc.metadata["index"]] for doc in relevant_documents]
        
        # Add human_feedback tool if not in the list
        if tools.human_feedback not in relevant_tools:
            relevant_tools.append(tools.human_feedback)

        if self.debug_mode:
            print("Suggested tools:")
            for tool in relevant_tools:
                print("Tool name:", tool.name)
                
        return relevant_tools

    def _load_prompts(self, filepath):
        # Helper function to load prompts from a given file
        messages = []
        temp_content = ""
        
        with open(filepath) as f:
            lines = f.readlines()
            turn = 0
            for line in lines:
                if line.startswith("system:"):
                    turn += 1
                    temp_content = line[7:].strip()
                elif line.startswith("input:"):
                    turn += 1
                    if turn == 2:
                        messages.append(SystemMessage(content=temp_content))
                    else:
                        messages.append(AIMessage(content=temp_content))
                    temp_content = line[6:].strip()
                elif line.startswith("output:"):
                    turn += 1
                    messages.append(HumanMessage(content=temp_content))
                    temp_content = line[7:].strip()
                else:
                    temp_content += line.strip()
            # Append the last message based on the turn value
            if turn == 1:
                messages.append(SystemMessage(content=temp_content))
            else:
                messages.append(AIMessage(content=temp_content))
                
        return messages

