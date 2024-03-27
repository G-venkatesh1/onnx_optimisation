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



def run_inference(model_path, json_file_path, npy_file_path):
    ort_session_1 = ort.InferenceSession(model_path)
    org_outputs = [x.name for x in ort_session_1.get_outputs()]
    model = onnx.load(model_path)
    input_dtype = get_input_dtype(model)
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
    if input_dtype == np.float32:
        a_INPUT = np.random.uniform(low=0.0, high=0.1, size=input_shape).astype(np.float32)
    elif input_dtype == np.float16:
        a_INPUT = np.random.uniform(low=0.0, high=0.1, size=input_shape).astype(np.float16)
    elif input_dtype == np.int8:
        a_INPUT = np.random.randint(low=-128, high=127, size=input_shape).astype(np.int8)

    # Get the output names for the modified model
    outputs = [x.name for x in ort_session.get_outputs()]

    # Run inference
    ort_outs = ort_session.run(outputs, {input_name: a_INPUT})

    # Map outputs to their names
    ort_outs_dict = dict(zip(outputs, ort_outs))
    # Create a dictionary to store layer names and their contents
    output_content = {layer_name: ort_outs_dict[layer_name].tolist() for layer_name in ort_outs_dict}

    # Get the output names for the modified model
    output_nodes_traversal = len(outputs)
    total_nodes_from_formula = len(model.graph.node)
    unique_op_types = list(set([node.op_type for node in model.graph.node]))
    all_node_names = [node.name for node in model.graph.node]


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


#!python int_layers.py /kaggle/input/fp_16_inception_v3/onnx/onnx/1/inspection_v3_fp16.onnx layer_node_stats_fp_16_inception_v3_final.json outputs_fp_16_inception_v3_final.npy