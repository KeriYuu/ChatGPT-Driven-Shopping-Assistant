import os
import configparser

class GlobalVariables:
    def __init__(self):
        self._global_dict = {}

    def set_value(self, key, value):
        self._global_dict[key] = value

    def get_value(self, key, default_value=None):
        try:
            return self._global_dict[key]
        except KeyError:
            print("No such key!")
            return default_value

def load_configuration_keys():
    config_path = "keys.txt"
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def construct_headers(rapid_host="amazon23.p.rapidapi.com/product-search"):
    config = load_configuration_keys()
    headers = {
        "X-RapidAPI-Key": config["RapidAPI"]["RAPIDAPI_API_KEY"],
        "X-RapidAPI-Host": rapid_host
    }
    return headers

def set_env():
    config = load_configuration_keys()
    os.environ["OPENAI_API_KEY"] = config["OpenAI"]["OPENAI_API_KEY"]
    os.environ["SERPAPI_API_KEY"] = config["SERP"]["SERPAPI_API_KEY"]
    os.environ["WOLFRAM_ALPHA_APPID"] = config["WolframAlpha"]["WOLFRAM_ALPHA_APPID"]

def read_funcs(filepath):
    function_names = []
    with open(filepath, encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("def ") and not line[4] == "_":
                function_name = line[4:].split("(")[0].strip()
                function_names.append(function_name)
    return function_names
