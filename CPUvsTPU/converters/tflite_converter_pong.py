import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
import sys

saved_dir = sys.argv[1]
tflite_dir = sys.argv[2]

#model = tf.keras.models.load_model(h5_dir, custom_objects={'_initializer':glorot_uniform()}, compile=False)
#converter = tf.lite.TFLiteConverter.from_saved_model("./foo")
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_dir, input_arrays=['default_policy/obs'], output_arrays=['default_policy/model/conv_out/BiasAdd'])
#converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]
tflite_model = converter.convert()
open(tflite_dir, "wb").write(tflite_model)
