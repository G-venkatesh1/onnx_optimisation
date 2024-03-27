import argparse
import json

def calculate_differences(json1, json2, output_json):
    # Load the two JSON files
    with open(json1, 'r') as file:
        data1 = json.load(file)

    with open(json2, 'r') as file:
        data2 = json.load(file)

    # Extract all "name" keys from both JSON files
    names1 = set(data1[i]['name'] for i in range(len(data1)))
    names2 = set(data2[i]['name'] for i in range(len(data2)))

    # Calculate differences for names in both JSON files
    differences = {}
    for name in names1.union(names2):
        dur1 = next((node['dur'] for node in data1 if node['name'] == name), 0)
        dur2 = next((node['dur'] for node in data2 if node['name'] == name), 0)
        differences[name] = abs(dur1 - dur2)

    # Save the differences to the output JSON file
    with open(output_json, 'w') as file:
        json.dump(differences, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate duration differences for every "name" in two JSON files.')
    parser.add_argument('json1', type=str, help='Path to the first JSON file')
    parser.add_argument('json2', type=str, help='Path to the second JSON file')
    parser.add_argument('output_json', type=str, help='Path to the output JSON file')
    args = parser.parse_args()

    calculate_differences(args.json1, args.json2, args.output_json)


#python diff.py E:\Mcw\onnx_optimisation\fp_32.json_2024-03-26_18-00-48.json E:\Mcw\onnx_optimisation\int_8.json_2024-03-26_18-01-30.json diff_result.json