import sys

dataset_dir = sys.argv[1]
checkpoint_dir = sys.argv[2]
tflite_dir= sys.argv[3]

#inputs = []
import numpy as np
#with open(dataset_dir, 'rb') as f:
#    for _ in range(500):
#        inputs.append(np.load(f))
#print("20 loaded image: ", inputs[20])
import tensorflow as tf 
def representative_data_gen():
    for one_hot in np.eye(500):
        yield[tf.dtypes.cast(one_hot, tf.float32)]

converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(checkpoint_dir, input_arrays=['default_policy/obs'], output_arrays=['default_policy/model/fc_out/BiasAdd'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant= converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input:', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output:', output_type)

open(tflite_dir, "wb").write(tflite_model_quant)

