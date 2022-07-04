import sys
import tensorflow as tf
checkpoint_dir = sys.argv[1]
tflite_dir= sys.argv[2]

images = []
import numpy as np

converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(checkpoint_dir, input_arrays=['default_policy/obs'], output_arrays=['default_policy/model/conv_out/BiasAdd'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.float16]
tflite_model_quant= converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input:', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output:', output_type)

open(tflite_dir, "wb").write(tflite_model_quant)

