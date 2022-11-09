from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations

for num_filters in range(32, 702+1, 10):
    # w^2 * 3*2 es aplicar un filtro sobre un canal. Hay que hacerlo tantas veces como canales de entrada para cada canal de salida. El +1 es por los bias
    #num_MACs = W * W * 3 * 3 * ((3+1)*num_filters + (num_filters+1)*num_filters*(L-1))
    #print("num_MACs:", num_MACs, "num_filters:", num_filters, "input_shape:", input_shape)
    model = Sequential()
    model.add(layers.Conv2D(num_filters, (3,3), padding='same', input_shape=(64,64,3), activation='relu'))
    for _ in range(4):
      model.add(layers.Conv2D(num_filters, (3,3), padding='same', activation='relu'))
    print(model.summary())
    input("continuar")
