import sys

sys.path.append("../build")
sys.path.append("../../../3rd_party/pyav/")
sys.path.append("../python_sdk")
import engine
import json


def unicode_convert(input):
    if isinstance(input, dict):
        return {unicode_convert(key): unicode_convert(value) for key, value in input.items()}
    elif isinstance(input, list):
        return [unicode_convert(element) for element in input]
    elif sys.version_info.major == 2 and isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input


def test_simple_decode():
    config_file = "../files/graph.json"
    with open(config_file, mode='r') as f:
        config = json.dumps(unicode_convert(json.loads(f.read())))
    print(config)
    graph = engine.Graph(config_file)
    print("graph init success")
    graph.start()
    print("graph start")
    # time.sleep(2)
    graph.close()


if __name__ == '__main__':
    test_simple_decode()
