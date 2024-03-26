import argparse
import onnx
import onnxruntime as ort
import numpy as np
import json

def get_input_dtype(model):
    """
    Get the data type of the model's input tensor.
    """
    input_type = model.graph.input[0].type.tensor_type.elem_type
    if input_type == onnx.TensorProto.FLOAT:
        return np.float32
    elif input_type == onnx.TensorProto.FLOAT16:
        return np.float16
    elif input_type == onnx.TensorProto.INT8:
        return np.int8
    else:
        raise ValueError("Unsupported input data type")

def add_all_nodes_as_output(model, ort_session):
    """
    Add all nodes as output nodes to the existing session.
    """
    # Get original output names
    org_outputs = [x.name for x in ort_session.get_outputs()]

    # Add all nodes as output nodes
    for node in model.graph.node:
        for output in node.output:
            if output not in org_outputs:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    # Serialize the modified model and create a new session
    modified_model = onnx.load_from_string(onnx.helper.serialize_model(model))
    modified_session = ort.InferenceSession(modified_model.SerializeToString(), providers=['CPUExecutionProvider'])

    return modified_session

def run_inference(model_path, json_file_path, npy_file_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Get the data type of the model's input tensor
    input_dtype = get_input_dtype(model)

    # Load the ONNX model with specified data type
    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'], 
                                        providers_options=[{'cpu_execution_provider': {'omp_num_threads': '1'}}], 
                                        sess_options=ort.SessionOptions())
    input_info = ort_session.get_inputs()[0]
    input_shape = input_info.shape

    # Generate random input data according to the data type
    if input_dtype == np.float32:
        a_INPUT = np.random.uniform(low=0.0, high=0.1, size=input_shape).astype(np.float32)
    elif input_dtype == np.float16:
        a_INPUT = np.random.uniform(low=0.0, high=0.1, size=input_shape).astype(np.float16)
    elif input_dtype == np.int8:
        a_INPUT = np.random.randint(low=-128, high=127, size=input_shape).astype(np.int8)

    # Add all nodes as output nodes to the existing session
    modified_session = add_all_nodes_as_output(model, ort_session)

    # Get the output names for the modified model
    outputs = [x.name for x in modified_session.get_outputs()]
    output_nodes_traversal = len(outputs)
    total_nodes_from_formula = len(model.graph.node)
    unique_op_types = list(set([node.op_type for node in model.graph.node]))
    all_node_names = [node.name for node in model.graph.node]

    # Run inference
    ort_outs = modified_session.run(outputs, {input_info.name: a_INPUT})

    # Map outputs to their names
    ort_outs_dict = dict(zip(outputs, ort_outs))

    # Create a dictionary to store layer names and their contents
    output_content = {layer_name: ort_outs_dict[layer_name].tolist() for layer_name in ort_outs_dict}

    # Save total_nodes_from_formula, output_nodes_traversal, unique_op_types, and all_node_names to JSON
    json_data = {
        "total_nodes_from_formula": total_nodes_from_formula,
        "output_nodes_traversal": output_nodes_traversal,
        "unique_op_types": unique_op_types,
        "all_node_names": all_node_names
    }
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    # Save intermediate layer outputs to .npy
    np.save(npy_file_path, output_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ONNX inference and store intermediate layer outputs and names in a JSON/NPY file.')
    parser.add_argument('model_path', type=str, help='Path to the ONNX model file')
    parser.add_argument('json_file_path', type=str, help='Path to the JSON file where the intermediate layer names will be stored')
    parser.add_argument('npy_file_path', type=str, help='Path to the npy file where the intermediate layer outputs will be stored')
    args = parser.parse_args()

    run_inference(args.model_path, args.json_file_path, args.npy_file_path)
