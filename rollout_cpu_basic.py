import tflite_runtime.interpreter as tflite
import time
import numpy as np
import statistics
import tensorflow as tf

def execute(model_path, threads, steps=2):
  interpreter = tflite.Interpreter(model_path=model_path, num_threads=threads)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]
  print(input_details)
  print(output_details)
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
