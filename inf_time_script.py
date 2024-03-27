import argparse
import onnxruntime as rt
import torch
import os

def load_model_and_run_inference(model_path, output_log):
    # Load the ONNX model
    sess_options = rt.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.profile_file_prefix = output_log  # Set the output log file name
    sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=sess_options)

    # Get input shape from the model
    input_shape = sess.get_inputs()[0].shape
    input_shape = tuple(d if d else 1 for d in input_shape)  # Replace None dimensions with 1
    print("Input shape:", input_shape)

    # Generate a random input tensor based on the input shape
    img = torch.rand(*input_shape)
    img_np = img.cpu().numpy()

    # Run inference
    pred = sess.run(None, {sess.get_inputs()[0].name: img_np})
    sess.end_profiling()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load an ONNX model and run inference with profiling.')
    parser.add_argument('model_path', type=str, help='Path to the ONNX model file')
    parser.add_argument('--output_log', type=str, default='onnxruntime_profile.json', help='Output log file name')
    args = parser.parse_args()

    # Ensure that the output log file has the correct extension
    if not args.output_log.endswith('.json'):
        args.output_log += '.json'

    load_model_and_run_inference(args.model_path, args.output_log)

#python E:/Mcw/onnx_optimisation/t1.onnx --output_log=fp_32.json