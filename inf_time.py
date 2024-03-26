from mlprodict.onnxrt import OnnxInference
import numpy as np
import pandas as pd
import json

# Load the ONNX model
onx_model_path = "/kaggle/input/sample_given/onnx/t1/1/t1.onnx"  # Replace "your_model.onnx" with the actual file path
oinf = OnnxInference(onx_model_path)

# Sample input data
sample_input_data = np.random.rand(1,10,10).astype(np.float32)  # Replace with your sample input data

# Run inference with sample input data and measure node time
res = oinf.run({'input': sample_input_data}, node_time=True)

# Convert the results to a DataFrame
df = pd.DataFrame(res[1])

# Specify the path to the JSON file where you want to store the result
json_file_path = 'execution_result.json'

# Write the execution result to the JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(df.to_dict(orient='records'), json_file, indent=4)

print(f"Execution result stored in '{json_file_path}'")

