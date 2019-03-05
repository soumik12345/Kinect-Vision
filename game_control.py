import json

def get_config_data():
    json_file = open('./config.json')
    json_string = json_file.read()
    json_data = json.loads(json_string)
    return json_data