import os
import keras
from matplotlib import pyplot
from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.constraints import maxnorm
import tensorflow as tf
import time
import logging
import subprocess

channels = 500

# Cargamos los datos
(X_train, _), (_, _) = cifar10.load_data()
X_train = X_train.astype('float32')
X_train = X_train / 255.0

def convert_to_TFLite(model_file_prefix, keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(model_file_prefix + ".tflite", "wb").write(tflite_model)

def predictTFLite(model_file, steps = 1):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    print("Input shape", input_shape)
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    output_predicted = []
    total_time = 0
    for _ in range(steps):
        interpreter.set_tensor(input_details[0]['index'], np.array(input_data))
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = (time.perf_counter() - start) * 1000
        total_time += inference_time
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_predicted.append(np.argmax(output_data))
    avg_time = total_time/steps
    return output_data, avg_time

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
        yield [input_value]

def quantize(model_file_prefix, keras_model):       
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_data_gen

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()
    open(model_file_prefix + "_quant.tflite", "wb").write(tflite_model_quant)
    
    
model_sizes = []
times_inferences = []
for channel in range(32, 32+channels, 50):
    model = Sequential()
    # CONV => RELU => CONV => RELU => POOL => DROPOUT
    model.add(Conv2D(channel, (3, 3), padding='same',input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(channel, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # CONV => RELU => CONV => RELU => POOL => DROPOUT
    model.add(Conv2D(channel + 32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(channel + 32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # FLATTERN => DENSE => RELU => DROPOUT
    model.add(Flatten())
    #model.add(Dense(512))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    # a softmax classifier
    model.add(Dense(10))
    model.add(Activation('softmax'))
    #print(model.summary())
    
    model_file_prefix = f"models_size/model_size_channel{channel}"
    
    tflite_model = convert_to_TFLite(model_file_prefix = model_file_prefix, keras_model = model)

    #_, avg_time = predictTFLite(model_file_prefix + ".tflite", steps = 50)
    #times_inferences.append(avg_time)
    quantize(model_file_prefix, model)
    
    
    model_size_MB = os.path.getsize(model_file_prefix + "_quant.tflite")/(2**20)
    print("Model size:", model_size_MB)
    model_sizes.append(model_size_MB) 

    edge_tpu = f'edgetpu_compiler {model_file_prefix}_quant.tflite'
    subprocess.Popen(edge_tpu.split()).communicate()

    input("Continuar...")

    move_tpu_file = f'mv model_size_channel{channel}_quant_edgetpu.tflite {model_file_prefix}_quant_edgetpu.tflite'
    subprocess.Popen(move_tpu_file.split()).communicate()

    move_tpu_log = f'mv model_size_channel{channel}_quant_edgetpu.log {model_file_prefix}_quant_edgetpu.log'
    subprocess.Popen(move_tpu_log.split()).communicate()
