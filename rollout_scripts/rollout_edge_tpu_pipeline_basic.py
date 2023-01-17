import tflite_runtime.interpreter as tflite
from pycoral.adapters import common
import time
import numpy as np
import statistics
import tensorflow as tf
import pycoral.pipeline.pipelined_model_runner as pipeline
from pycoral.adapters import common

def make_runner(model_prefix, num_interpreters):
  interpreters = [tflite.Interpreter(model_path=model_prefix + f"_segment_{num_dev}_of_{num_interpreters}_edgetpu.tflite",
                  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1', options={"device": f":{num_dev}"})]) for num_dev in range(num_interpreters)]
  for interpreter in interpreters:
    interpreter.allocate_tensors()
  return pipeline.PipelinedModelRunner(interpreters)


def execute(model_prefix, num_segments, steps=2):
  runner = make_runner(model_prefix=model_prefix, num_interpreters=num_segments)

  size = common.input_size(runner.interpreters()[0])
  name = common.input_details(runner.interpreters()[0], 'name')
  input_shape = common.input_details(runner.interpreters()[0], 'shape')
  print("Input size:", size)
  print("Name:", name)
  print("Input shape:", input_shape)

  #input_val = tf.constant(1., shape=input_shape)
  input_val = np.full(input_shape, 1., dtype = np.float32)
  print(type(input_val))
  #input("continuar")
  total_time_ms = 0
  times = []
  for i in range(steps):
    start = time.perf_counter()
    #print(input_val.dtype)
    runner.push({name: input_val})
    output_val = runner.pop()
    inference_time = (time.perf_counter() - start) * 1000
    print("Output shape:", np.array(list(output_val.values())).shape)
    if i == 0:
      continue
    #total_time_ms += inference_time
    print("Inference time:", inference_time)
    times.append(inference_time)
  #return total_time_ms/(steps-1)
  return statistics.median(times)
