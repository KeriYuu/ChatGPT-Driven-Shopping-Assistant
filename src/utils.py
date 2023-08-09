import os
import configparser
class GOLOBAL_VAL():
    def _init():  
        global _global_dict
        _global_dict = {}


    def set_value(key, value):
        _global_dict[key] = value


    def get_value(key, defValue=None):
        try:
            return _global_dict[key]
        except KeyError:
            print("no such key!")

def readkey():
    config_path = "keys.txt"
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def construct_rapid_headers(rapid_host="amazon23.p.rapidapi.com/product-search"):
    config = readkey()
    headers = {
        "X-RapidAPI-Key": config["RapidAPI"]["RAPIDAPI_API_KEY"],
        "X-RapidAPI-Host": rapid_host
    }
    return headers

def set_env():
    config = readkey()
    os.environ["OPENAI_API_KEY"] = config["OpenAI"]["OPENAI_API_KEY"]
    os.environ["SERPAPI_API_KEY"] = config["SERP"]["SERPAPI_API_KEY"]
    os.environ["WOLFRAM_ALPHA_APPID"] = config["WolframAlpha"]["WOLFRAM_ALPHA_APPID"]

def read_funcs(path):
    func_names = []
    with open(path, encoding="utf-8") as f:
        line = f.readline()
        while line:
            if line.startswith("def ") and not line[4] == "_":
                line = line[4:]
                func_name = line.split("(")[0]
                func_names.append(func_name)
            line = f.readline()
    return func_names
