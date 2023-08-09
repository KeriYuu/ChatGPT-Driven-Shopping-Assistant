import re
from typing import List, Union, Optional, Any
from uuid import UUID
from retriever import Retriever
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler


class FormattedPrompt(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]
    cache: dict

    def format_messages(self, **kwargs) -> list:
        length = 9
        intermediate_steps = kwargs.pop("intermediate_steps")
        truncate_len = ((len(intermediate_steps) - length) // 3) * 3
        thoughts = ""

        for action, observation in intermediate_steps[max(truncate_len, 0):]:
            thoughts += action.log
            thoughts += f"\\nObservation: {observation}\\nThought: "

        summarize_history = ""
        if truncate_len > 0:
            llm = ChatOpenAI(temperature=0)
            with open("src/prompts/summarize_history.txt") as f:
                prefix = f.read()
            for i in range(truncate_len - 1):
                action_1, observation_1 = intermediate_steps[i]
                action_2, observation_2 = intermediate_steps[i + 1]
                history = f"Thought: {action_1.log}\\nObservation: {observation_1}\\nThought: {action_2.log}\\nObservation: {observation_2}\\n"
                if history not in self.cache.keys():
                    temp = llm([HumanMessage(content=prefix.format(content=history))]).content + "\\n"
                    self.cache[history] = temp
                    summarize_history += temp + "\\n" * 3
                else:
                    summarize_history += self.cache[history] + "\\n" * 3

        kwargs["chat_history"] = summarize_history + thoughts
        return super().format_messages(**kwargs)


class ProcessedOutputParser(AgentOutputParser):

    def process_output(self, output: Union[str, List[str]]) -> Any:
        return super().process_output(output)


class AgentInterface:

    def __init__(self, debug: bool = False, max_tools: int = 3):
        self.debug = debug
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True, verbose=self.debug)
        self.memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0.7), max_token_limit=1000, memory_key="chat_history")
        self.memory.output_key = "output"
        with open("src/prompts/template_with_history.txt") as f:
            self.template_with_history = f.read()
        self.retriever = Retriever(debug=self.debug, max_tools=max_tools)
        self.output_parser = ProcessedOutputParser()
        self.show = []

    def initiate(self):
        while True:
            user_input = input("Your question here:\\n")
            self.execute(user_input)

    def execute(self, user_input):
        tools = self.retriever.retrieve_tools(user_input)
        prompt_with_history = FormattedPrompt(
            template=self.template_with_history,
            tools=tools,
            cache={},
            input_variables=["input", "intermediate_steps", "chat_history"]
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=prompt_with_history, verbose=self.debug)
        tool_names = [tool.name for tool in tools]
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            stop=["Observation:", "observation:"],
            output_parser=self.output_parser,
            allowed_tools=tool_names,
            verbose=self.debug,
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent,
                                                                 tools=tools, verbose=True, memory=self.memory,
                                                                 return_intermediate_steps=True, max_iterations=100, callbacks=[BaseCallbackHandler()])
        response = self.agent_executor(user_input)
        return response

    def retrieve_steps(self):
        travel_plan = ""
        for action, observation in self.agent_executor.intermediate_steps:
            travel_plan += f"Thought:\\n{action.log}\\nObservation:\\n{observation}\\n\\n"
        return travel_plan

