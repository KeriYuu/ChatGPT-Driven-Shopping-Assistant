import time
from langchain.tools import tool
from  utils import construct_rapid_headers
import json
import requests

from utils import GLOBAL_VAL

@tool
def recommend_products(shopping_info: str) -> str:
    """
    The description is as follows:
    args:
        -shopping_info: The shopping_info argument is a dictionary that contains the product query and 
        the country. This argument must be in the following format:
        {"query": "xxx", "country": "xxx"}
        Make sure that country should be one of this : "US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP", it is very important!!!
    return:
        a list of products' information
    functionality:
        This tool will return a list of products' information if you need to search for a product, you can call this tool.
    usage example:
        Action: recommend_products
        Action Input: {"query": "xxx", "country": "xxx"}
    composition instructions:
        Before you call this tool, you must call human_feedback to make sure the product and country to query, and the country should be one of "US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP".
        After you call this tool, you need to call human_feedback tool to choose a product.

    """
    shopping_dict = json.loads(shopping_info)
    url = "https://real-time-amazon-data.p.rapidapi.com/search"

    querystring = {"query": shopping_dict["query"], "country": shopping_dict["country"]}

    headers = construct_rapid_headers('real-time-amazon-data.p.rapidapi.com')

    response = requests.get(url, headers=headers, params=querystring)
    all_products = "\n"
    for product in response.json()["data"]["products"][:5]:
        product_description = product["product_title"] + \
                            "\nOrigin Price: " + str(product["product_original_price"]) + \
                            "\nPrice: " + str(product["product_price"]) + \
                            "\nRating" + str(product["product_star_rating"]) + \
                            "\nURL: " + product["product_url"] + "\n"
        all_products = all_products + product_description
    return all_products.strip()

@tool
def human_feedback(question: str) -> str:
    """The description is as follows:
    args:
        -question: the question that you want to ask for more personal opinion
    return:
        user's answer
    functionality:
        This tool can be used to acquire the user's choice and personal preference
    usage example:
        Action: human_feedback
        Action Input: xxx
    """
    global local_chat_counter
    global_value = GLOBAL_VAL()
    global_value.set_value("question", question)
    global_value.set_value("get_user_input", True)
    global_value.set_value("wait_for_LLM", False)
    while (global_value.get_value("get_user_input")):
        time.sleep(0.1)
    return global_value.get_value("inputs")
