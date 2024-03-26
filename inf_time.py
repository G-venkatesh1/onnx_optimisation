# from mlprodict.onnxrt import OnnxInference
# import numpy as np
# import pandas as pd
# import json

# # Load the ONNX model
# onx_model_path = "/kaggle/input/sample_given/onnx/t1/1/t1.onnx"  # Replace "your_model.onnx" with the actual file path
# oinf = OnnxInference(onx_model_path)

# # Sample input data
# sample_input_data = np.random.rand(1,10,10).astype(np.float32)  # Replace with your sample input data

# # Run inference with sample input data and measure node time
# res = oinf.run({'input': sample_input_data}, node_time=True)

# # Convert the results to a DataFrame
# df = pd.DataFrame(res[1])

# # Specify the path to the JSON file where you want to store the result
# json_file_path = 'execution_result_sample.json'

# # Write the execution result to the JSON file
# with open(json_file_path, 'w') as json_file:
#     json.dump(df.to_dict(orient='records'), json_file, indent=4)

# print(f"Execution result stored in '{json_file_path}'")

import argparse
import onnx
import onnxruntime as ort
import numpy as np
import json

def run_inference(model_path, input_shape, input_name, json_file_path):
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

    # Get the output names for the modified model
    outputs = [x.name for x in ort_session.get_outputs()]

    # Generate random input data
    input_data = np.random.rand(*input_shape).astype(np.float32)

    # Run inference
    ort_outs = ort_session.run(outputs, {input_name: input_data})

    # Map outputs to their names
    ort_outs_dict = dict(zip(outputs, ort_outs))
    # Create a dictionary to store layer names and their contents
    output_content = {layer_name: ort_outs_dict[layer_name].tolist() for layer_name in ort_outs_dict}

    # Save content to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(output_content, json_file)

    # Print shapes of the intermediate layer outputs
    for layer_name, lay_wise_output in ort_outs_dict.items():
        print(f"Shape of output '{layer_name}': {lay_wise_output.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ONNX inference and store intermediate layer outputs in a JSON file.')
    parser.add_argument('model_path', type=str, help='Path to the ONNX model file')
    parser.add_argument('input_shape', type=int, nargs='+', help='Shape of the input data')
    parser.add_argument('input_name', type=str, help='Name of the input to the ONNX model')
    parser.add_argument('json_file_path', type=str, help='Path to the JSON file where the intermediate layer outputs will be stored')
    args = parser.parse_args()

    input_shape = tuple(args.input_shape)
    run_inference(args.model_path, input_shape, args.input_name, args.json_file_path)

