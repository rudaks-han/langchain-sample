from pprint import pprint

data = {
    "name": "Alice",
    "age": 30,
    "interests": ["reading", "coding", "hiking"],
    "address": {"city": "Wonderland", "zip": "12345"},
}

pprint(data, width=50)

nested_data = {"level1": {"level2": {"level3": {"key": "value"}}}}

pprint(nested_data, depth=2)
