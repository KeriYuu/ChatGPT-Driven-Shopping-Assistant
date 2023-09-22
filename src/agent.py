from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, load_tools
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from retriever import Retriever
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
import re
import json

class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]
    cache: dict

    def format_messages(self, **kwargs) -> list:
        length = 9
        intermediate_steps = kwargs["intermediate_steps"]
        truncate_len = ((len(intermediate_steps) - length) // 3) * 3
        thoughts = ""

        for action, observation in intermediate_steps[max(truncate_len, 0):]:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        memories = ""
        if truncate_len > 0:
            llm = ChatOpenAI(temperature=0)
            with open("prompts/prompting.json") as f:
                data = json.load(f)['memories']
                prefix = data["description"] + "\n" + "\n".join(data["information_format"]) + "\n\n" + \
                         "\n".join(data["job"]) + "\n" + data["prompt"]
            for i in range(truncate_len - 1):
                action_1, observation_1 = intermediate_steps[i]
                action_2, observation_2 = intermediate_steps[i + 1]
                history = f"Thought: {action_1.log}\nObservation: {observation_1}\n\n" \
                          f"Thought: {action_2.log}\nObservation: {observation_2}\n\n"
                if history not in self.cache.keys():
                    temp = llm([HumanMessage(content=prefix.format(content=history))]).content + "\n"
                    self.cache[history] = temp
                    memories += temp + "\n" * 3
                else:
                    memories += self.cache[history] + "\n" * 3

        kwargs["chat_history"] = kwargs["chat_history"] + memories
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]



class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(

                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentAction(tool="None", tool_input="None", log=llm_output)
        action = match.group(1).strip()
        action_input = match.group(2)

        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class Agent:
    def __init__(self, debug=False, max_tools=10):
        self.debug = debug
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True, verbose=self.debug)

        self.memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0.7), max_token_limit=1000, memory_key="chat_history")
        self.memory.output_key = "output"

        with open("prompts/templates.txt") as f:
            self.templates = f.read()
        self.retriever = Retriever(debug=self.debug, max_tools=max_tools)
        self.output_parser = CustomOutputParser()
        self.show = []

    def run(self, user_input):
        tools = self.retriever.get_tools(user_input)
        tools += load_tools(["serpapi", "llm-math"], llm=self.llm)
        prompt_with_history = CustomPromptTemplate(
            template=self.templates,
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
        return_intermediate_steps=True, max_iterations=100)

        response = self.agent_executor(user_input)
        return response

    def get_intermediate_steps(self):
        travel_plan = ""
        for action, observation in self.agent_executor.intermediate_steps:
            travel_plan += f"Thought:\n{action.log}\nObservation:\n{observation}\n\n"
        return travel_plan

    def command_start(self):
        while True:
            user_input = input("Your question here:\n")
            self.run(user_input)
