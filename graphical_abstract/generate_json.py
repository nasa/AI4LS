import json

# This is the data you would normally read from a JSON file
experiment_data = {
    "experiment_groups": [
        {"group": "1-10", "mice": 120, "gender": "female", "strand": "B6CF1", "dose": 2.6},
        # ... add other groups as necessary
    ]
}

# Writing the test data to a JSON file for demonstration purposes
with open('./experiment_data.json', 'w') as f:
    json.dump(experiment_data, f)

