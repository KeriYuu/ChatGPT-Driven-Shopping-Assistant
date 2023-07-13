import gradio as gr
import time
import threading
from utils import GOLOBAL_VAL,set_env
from agent import Agent

set_env()
local_chat_counter = 0
# init global value
global_value = GOLOBAL_VAL()
global_value.set_value('chat_counter', 0)
global_value.set_value('get_user_input', True)


shopping_agent = Agent(debug=False, max_tools=10)
global_value.set_value("shopping_agent", shopping_agent)

def start_agent(inputs):
    response = shopping_agent.run(inputs)
    return response


def visualize_plan():
    return shopping_agent.get_intermediate_steps()

def handle_inputs_submit(inputs, chatbot):
    global_value.set_value("chat_counter", global_value.get_value("chat_counter") + 1)
    if global_value.get_value("chat_counter") == 1:
        threads = [threading.Thread(target=start_agent, kwargs={"inputs": inputs})]
        for t in threads:
            t.start()
    global_value.set_value("inputs", inputs)
    global_value.set_value("get_user_input", False)
    global_value.set_value("wait_for_LLM", True)

    # wait for LLM query
    while (global_value.get_value("wait_for_LLM")):
        time.sleep(0.1)

    chatbot.append([inputs, global_value.get_value("question")])
    return chatbot


# Resetting to blank
def reset_textbox():
    global_value.set_value('chat_counter', 0)
    bot_msg = "I'm a shopping assistant. Please describe your shopping needs or purpose."
    chatbot = [[None, ""]]
    for character in bot_msg:
        chatbot[-1][1] += character
        time.sleep(0.01)
        yield chatbot
    return chatbot


title = """<h1 align="center">Save Time with Your Personal Online Shopping Assistant</h1>"""


with gr.Blocks(
        css="""#col_container { margin-left: auto; margin-right: auto;} #chatbot {height: 400px; overflow: auto;}""",
        theme='abidlabs/font-test') as demo:
    gr.HTML(title)
    gr.HTML("""<h3 align="center">🌟 Please state your shopping needs or product usage</h3>""")
    with gr.Column(elem_id="col_container"):
        with gr.Row():
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(label='Shopping Assistant', elem_id="chatbot")
                inputs = gr.Textbox(placeholder="Interact with your shopping assistant.", label="Type an input and press Enter")
                gr.Examples(
                            examples=[["I plan to go hiking on the weekend."], ["My budget is only $100"], ["I like very light weight clothing."],
                            ["I have UV allergies."]], inputs=inputs)
                state = gr.State([])
            with gr.Column(scale=5):
                with gr.Accordion(label="Your shopping plan:", open=True):
                    whole_plan = gr.Textbox(label="Generated in interaction", interactive=False)   
                with gr.Row(scale=2):  
                    b1 = gr.Button(value="Restart")
                    b2 = gr.Button(value="See the whole plan")
                with gr.Accordion("OpenAI Key", open=False):
                    openai_gpt4_key = gr.Textbox(label="OpenAI Key", value="", type="password", placeholder="sk..",
                                                info="You have to provide your own keys for this app to function properly", )
                with gr.Accordion("Parameters", open=False):
                    top_p = gr.Slider(minimum=-0, maximum=1.0, value=0.7, step=0.05, interactive=True,
                                    label="Top-p (nucleus sampling)", )
                    temperature = gr.Slider(minimum=-0, maximum=5.0, value=1.0, step=0.1, interactive=True,
                                            label="Temperature", )
                    chat_counter = gr.Number(value=0, visible=False, precision=0)


    inputs.submit(handle_inputs_submit, [inputs, chatbot], [chatbot])  # openai_api_key
    b1.click(reset_textbox, [], [chatbot])
    b2.click(visualize_plan, [], [whole_plan])

demo.queue(max_size=99, concurrency_count=20).launch(debug=True, share=True)
