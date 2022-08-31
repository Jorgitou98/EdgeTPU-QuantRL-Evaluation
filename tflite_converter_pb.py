import tensorflow as tf
import sys

saved_dir = sys.argv[1]
tflite_dir = sys.argv[2]

converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_dir, input_arrays=['default_policy/obs'], output_arrays=['default_policy/model/conv_out/BiasAdd'])
#converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]
tflite_model = converter.convert()
open(tflite_dir, "wb").write(tflite_model)
