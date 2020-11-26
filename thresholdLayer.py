import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
np.random.seed(seed=1)
tf.set_random_seed(seed=1)
true_cutoff = 0.2
n=10000
x = np.random.rand(n).reshape(-1,1)
y = x>=true_cutoff
# plot random numbers:
# import matplotlib.pyplot as plt
# plt.scatter(x, y)
# plt.show()

input_layer = keras.Input(shape=(1,))
class ThresholdLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ThresholdLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name="threshold", shape=(1,), initializer="uniform",
                                      trainable=True)
        super(ThresholdLayer, self).build(input_shape)
    def call(self, x):
        return keras.backend.sigmoid(100*(x-self.kernel))

def compute_output_shape(self, input_shape):
        return input_shape
out = ThresholdLayer()(input_layer)
model = keras.Model(inputs=input_layer, outputs=out)
model.compile(optimizer="sgd", loss="mse")
model.fit(x, y, epochs=50)

threshold = model.get_weights()
print("the trained weights:", threshold)
