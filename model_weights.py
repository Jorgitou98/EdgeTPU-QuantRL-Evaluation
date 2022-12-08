import tflite_runtime.interpreter as tflite
import numpy as np
import math
tflite_interpreter = tflite.Interpreter(model_path='checkpoints/ppo/model2/exported_models/checkpoint-51_quant.tflite')
tflite_interpreter.allocate_tensors()

tensor_details = tflite_interpreter.get_tensor_details()
tensors = np.array([])
weights_means = []
for dict in tensor_details:
    i = dict['index']
    tensor_name = dict['name']
    scales = dict['quantization_parameters']['scales']
    zero_points = dict['quantization_parameters']['zero_points']
    #if tensor_name[-6::] != "Conv2D":
    #  continue
    tensor = tflite_interpreter.tensor(i)()
    print(scales)
    print(tensor.shape)
    input("parar")
    tensors = np.concatenate((tensors, tensor.flatten()))
    #tensors.append(tensor.flatten())
    weights_means.append(np.mean(tensor))
    #print(i, tensor_name, scales, zero_points, tensor)
    range = np.max(tensor)-np.min(tensor)
    print(f"Range {i}:", np.max(tensor)-np.min(tensor))
    print(f"Mean {i}:", np.std(tensor))
    '''
    See note below
    '''
#print(max(weights_means))
print(max(abs(np.mean(tensor) - np.std(tensor)), abs(np.mean(tensor) + np.std(tensor))))
