import argparse
import onnx
import onnxruntime as ort
import numpy as np
import json

def get_node_stats(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Get the total number of nodes
    total_nodes = len(model.graph.node)
    node_stats = {"total_nodes": total_nodes}

    # List down all unique operation types
    unique_op_types = list(set([node.op_type for node in model.graph.node]))
    node_stats["unique_op_types"] = unique_op_types

    # Print the node stats
    print("Node Stats:", node_stats)

def run_inference(model_path, json_file_path):
    # Load the ONNX model
    ort_session_1 = ort.InferenceSession(model_path)

    # Get the original output names
    org_outputs = [x.name for x in ort_session_1.get_outputs()]

    # Load the model
    model = onnx.load(model_path)

    # Add all layers as output
    for node in model.graph.node:
        for output in node.output:
            if output not in org_outputs:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    # Serialize the modified model and create a new session
    ort_session = ort.InferenceSession(model.SerializeToString())

    # Get input information
    input_info = ort_session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    # Generate random input data
    a_INPUT = np.random.uniform(low=0.0, high=0.1, size=input_shape).astype(np.float32)

    # Get the output names for the modified model
    outputs = [x.name for x in ort_session.get_outputs()]
    no_of_output_nodes = len(outputs)

    # Run inference
    ort_outs = ort_session.run(outputs, {input_name: a_INPUT})

    # Map outputs to their names
    ort_outs_dict = dict(zip(outputs, ort_outs))
    # Create a dictionary to store layer names and their contents
    output_content = {layer_name: ort_outs_dict[layer_name].tolist() for layer_name in ort_outs_dict}

    # Save content to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(output_content, json_file)

    # Print the number of output nodes
    print("No. of Output Nodes:", no_of_output_nodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ONNX inference and store intermediate layer outputs in a JSON file.')
    parser.add_argument('model_path', type=str, help='Path to the ONNX model file')
    parser.add_argument('json_file_path', type=str, help='Path to the JSON file where the intermediate layer outputs will be stored')
    args = parser.parse_args()

    # Run inference and get node stats
    run_inference(args.model_path, args.json_file_path)
    get_node_stats(args.model_path)
