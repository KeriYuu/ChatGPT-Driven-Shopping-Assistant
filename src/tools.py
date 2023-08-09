import time
from langchain.utilities import SerpAPIWrapper
from langchain.tools import tool, HumanInputRun
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from  utils import *
import http, requests
import os
import random
import json

@tool
def google_search(question: str):
    """The description is as follows:
    args:
        -question: the general question you want to ask
    return:
        search result that comes from google search engine
    functionality:
        This tool can be used to search for latest information that relates to a general question
    usage example:
        Action: google_search
        Action Input: xxx
    composition instructions:
        This tool is the last resort, if other tools can't help
    """
    config = load_configuration_keys()
    os.environ["SERPAPI_API_KEY"] = config["SERP"]["SERPAPI_API_KEY"]
    try:
        search = SerpAPIWrapper()
        res = search.run(question)
    except BaseException:
        res = "invalid response from google"
    return res


@tool
def wolfram_calculator(problem: str):
    """The description is as follows:
    args:
        -problem: the math problem you want to solve, try to use math symbol more
    return:
        the answer to the math problem
    functionality:
        This is a powerful calculator, You can use it to calculate large numbers and solve complex math problems
    usage example:
        Action: wolfram_calculator
        Action Input: xxx
    composition instructions:
        You should write the problem into math symbol format before you call this tool
    """
    config = load_configuration_keys()
    os.environ["WOLFRAM_ALPHA_APPID"] = config["WolframAlpha"]["WOLFRAMALPHA_APP_ID"]
    try:
        wolfram = WolframAlphaAPIWrapper()
        res = wolfram.run(problem)
    except BaseException:
        res = "invalid response from wolfram"
    return res

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
    def get_input() -> str:
            print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
            contents = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line == "q":
                    break
                contents.append(line)
            return "\n".join(contents)
    human_feedbacks = HumanInputRun(input_func=get_input)(question)
    print(human_feedbacks)
    return human_feedbacks

@tool
def recommend_products(shopping_info: str) -> str:
    """
    The description is as follows:
    args:
        -shopping_info: The shopping_info argument is a dictionary that contains the product query and 
        the country. This argument must be in the following format:
        {"query": "xxx", "country": "xxx"}
    return:
        a list of products' information
    functionality:
        This tool will return a list of products' information if you need to search for a product, you can call this tool.
    usage example:
        Action: recommend_products
        Action Input: {"query": "xxx", "country": "xxx"}
    composition instructions:
        Before you call this tool, you must call geographic_places to acquire the city's country, and call human_feedback to make sure the product to query.
        After you call this tool, you need to call human_feedback tool to choose a product.
    """
    try:
        shopping_dict = json.loads(shopping_info)
        url = "amazon23.p.rapidapi.com/product-search"

        querystring = {"query": shopping_dict["query"], "country": shopping_dict["country"]}

        headers = construct_headers("amazon23.p.rapidapi.com")

        response = requests.get(url, headers=headers, params=querystring)
        all_products = "\n"
        for product in response.json()["result"][:5]:
            product_description = product["title"] + \
                                "\nASIN: " + product["asin"] + \
                                "\nPrice: " + str(product["price"]["current_price"]) + " " + product["price"]["currency"] + \
                                "\nRating: " + str(product["reviews"]["rating"]) + \
                                "\nTotal reviews: " + str(product["reviews"]["total_reviews"]) + \
                                "\nURL: " + product["url"] + "\n"
            all_products = all_products + product_description
    except:
        return "invalid recommend_products tool, please try again"
    return all_products.strip()


@tool
def geographic_places(city_name: str) -> dict:
    """The description is as follows:
    args:
        -city_name: the name of your queried city
    return:
        a python-format dictionary that contains geographic information. In details, it contains 7 kinds of information: country of the queried city(country), timezone(timezone), name of the city(name), longitude(lon), latitude(lat), population(population) and the api call status(status).
    functionality:
        This tool can be used to search for geographic information about a city. If you want to get the longitude, latitude and other geographic information about a city, you can call this tool.
    usage example:
        Action: geographic_places
        Action Input: xxx
    composition instructions:
        You must call this tool to get the longitude and latitude before you call recommend_hotel and rental_car tools
    """
    try:
        url = "opentripmap-places-v1.p.rapidapi.com/en/places/geoname"

        querystring = {"name": city_name}
        headers = construct_headers("opentripmap-places-v1.p.rapidapi.com")
        response = requests.get(url, headers=headers, params=querystring)
    except BaseException as e:
        return str(e)
    return response.json()