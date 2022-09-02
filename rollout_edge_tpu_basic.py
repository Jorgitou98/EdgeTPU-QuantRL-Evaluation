import tflite_runtime.interpreter as tflite
from pycoral.adapters import common
import time
import numpy as np
import statistics
import tensorflow as tf

def execute(model_path, steps=2):
  interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0] 
  print(input_details)
  print(output_details)
  #input = np.array(np.random.random_sample(input_details[0]["shape"]), dtype=np.float32)
  #print(input.shape)
  #common.set_input(interpreter, input)
  total_time_ms = 0
  times = []
  for i in range(steps):
    interpreter.set_tensor(input_details['index'], tf.constant(1., shape=input_details['shape']))
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = (time.perf_counter() - start) * 1000
    #print("Output shape:", interpreter.get_tensor(output_details['index']).shape)
    if i == 0:
      continue
    #total_time_ms += inference_time
    print("Inference time:", inference_time)
    times.append(inference_time)
  #return total_time_ms/(steps-1)
  return statistics.median(times)
