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
import numpy as np
import pandas as pd
import json
from mlprodict.onnxrt import OnnxInference

def run_inference(model_path, input_data, input_name):
    # Load the ONNX model
    oinf = OnnxInference(model_path)
    
    # Run inference with input data and measure node time
    res = oinf.run({input_name: input_data}, node_time=True)
    
    # Convert the results to a DataFrame
    df = pd.DataFrame(res[1])
    
    return df

def main(model_path, input_data, input_name, json_file_path):
    # Run inference
    result_df = run_inference(model_path, input_data, input_name)

    # Write the execution result to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(result_df.to_dict(orient='records'), json_file, indent=4)

    print(f"Execution result stored in '{json_file_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ONNX inference and store result in a JSON file.')
    parser.add_argument('model_path', type=str, help='Path to the ONNX model file')
    parser.add_argument('input_data', type=float, nargs='+', help='Input data as a list of floats separated by spaces')
    parser.add_argument('input_name', type=str, help='Name of the input to the ONNX model')
    parser.add_argument('json_file_path', type=str, help='Path to the JSON file where the result will be stored')
    args = parser.parse_args()

    input_data = np.array(args.input_data, dtype=np.float32).reshape(1, -1)  # Reshape input data to (1, N)
    main(args.model_path, input_data, args.input_name, args.json_file_path)

