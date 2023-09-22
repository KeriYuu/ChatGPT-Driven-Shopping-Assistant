import os
import configparser
class GLOBAL_VAL:
    _instance = None  

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super(GLOBAL_VAL, cls).__new__(cls)
            cls._instance._global_dict = {}  
        return cls._instance 

    def set_value(self, key, value):
        """Set a key-value pair in the global dictionary."""
        self._global_dict[key] = value

    def get_value(self, key, defValue=None):
        """Get the value for a key from the global dictionary."""
        try:
            return self._global_dict[key]
        except KeyError:
            print("no such key!")
            return defValue


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
