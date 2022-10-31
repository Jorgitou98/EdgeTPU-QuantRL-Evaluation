
import tflite_runtime.interpreter as tflite
from pycoral.adapters import common
import time
import numpy as np
import statistics
import tensorflow as tf
from pycoral.adapters import common
import threading
import queue

def make_interpreters(model_prefix, num_interpreters):
  if num_interpreters > 1:
    interpreters = [tflite.Interpreter(model_path=f"{model_prefix}_seg{num_dev}of{num_interpreters}_quant_edgetpu.tflite",
                    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1', options={"device": f":{num_dev}"})]) for num_dev in range(num_interpreters)]
  for interpreter in interpreters:
    interpreter.allocate_tensors()
  return interpreters

def put_values_queue(first_queue, input_shape, batch):
  for _ in range(batch):
    first_queue.put(np.full(input_shape, 1., dtype = np.float32))

def execute(model_prefix, num_segments, steps=2, batch=1):
  interpreters = make_interpreters(model_prefix=model_prefix, num_interpreters=num_segments)

  #size = common.input_size(runner.interpreters()[0])
  #name = common.input_details(runner.interpreters()[0], 'name')
  input_shape = common.input_details(interpreters[0], 'shape')
  #print("Input size:", size)
  #print("Name:", name)
  print("Input shape:", input_shape)
  #print(type(input_val))

  queues = [queue.Queue(batch) for _ in range(num_segments+1)]

  def segment_inference(num_segment, batch):
    input_details = interpreters[num_segment].get_input_details()
    output_details = interpreters[num_segment].get_output_details()
    input_details[0]["dtype"] = np.int8
    for _ in range(batch):
      input_val = queues[num_segment].get()
      print(input_details[0], input_val.dtype)
      interpreters[num_segment].set_tensor(input_details[0]['index'], input_val)
      interpreters[num_segment].invoke()
      output = interpreters[num_segment].get_tensor(output_details[0]['index'])
      queues[num_segment+1].put(output)

  avgs_times = []
  for _ in range(steps):
    put_values_queue(first_queue = queues[0], input_shape = input_shape, batch = batch)
    inference_threads = [threading.Thread(target=segment_inference, args=(num_segment, batch)) for num_segment in range(num_segments)]
    start = time.perf_counter()
    for inference_thread in inference_threads:
      inference_thread.start()
    for inference_thread in inference_threads:
      inference_thread.join()
    average_time_ms = (time.perf_counter() - start) / batch * 1000
    print("Time:", average_time_ms)
    avgs_times.append(average_time_ms)
    while not queues[-1].empty():
      print("Final output:", queues[-1].get())
    #input("continuar")
  return statistics.median(avgs_times)
