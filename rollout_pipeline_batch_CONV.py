import tflite_runtime.interpreter as tflite
from pycoral.adapters import common
import time
import numpy as np
import statistics
import tensorflow as tf
import pycoral.pipeline.pipelined_model_runner as pipeline
from pycoral.adapters import common
import threading

def make_runner(model_prefix, num_interpreters):
  if num_interpreters > 1:
    interpreters = [tflite.Interpreter(model_path=f"{model_prefix}_segment_{num_dev}_of_{num_interpreters}_edgetpu.tflite",
                    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1', options={"device": f":{num_dev}"})]) for num_dev in range(num_interpreters)]
  else:
    interpreters = [tflite.Interpreter(model_path=f"{model_prefix}_edgetpu.tflite",
                  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1', options={"device": f":{num_dev}"})]) for num_dev in range(num_interpreters)]
  for interpreter in interpreters:
    interpreter.allocate_tensors()
  return pipeline.PipelinedModelRunner(interpreters)


def execute(model_prefix, num_segments, steps=2, batch_size=1):
  runner = make_runner(model_prefix=model_prefix, num_interpreters=num_segments)

  size = common.input_size(runner.interpreters()[0])
  name = common.input_details(runner.interpreters()[0], 'name')
  input_shape = common.input_details(runner.interpreters()[0], 'shape')
  print("Input size:", size)
  print("Name:", name)
  print("Input shape:", input_shape)

  input_val = np.full(input_shape, 1., dtype = np.float32)
  print(type(input_val))

  def producer():
    for _ in range(batch_size):
      runner.push({name: input_val})
    runner.push({})

  def consumer():
    while True:
      result = runner.pop()
      if not result:
        break
  avgs_times = []
  for _ in range(steps):
    runner = make_runner(model_prefix=model_prefix, num_interpreters=num_segments)
    start = time.perf_counter()
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()
    average_time_ms = (time.perf_counter() - start) / batch_size * 1000
    print("Time:", average_time_ms)
    avgs_times.append(average_time_ms)
  return statistics.median(avgs_times)
