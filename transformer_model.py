import tensorflow as tf
from tensorflow.keras import layers, models

class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__()
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weights_linear',
                                              shape=(input_shape[-1], 1),
                                              initializer='uniform',
                                              trainable=True)
        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(1,),
                                           initializer='uniform',
                                           trainable=True)
        self.weights_periodic = self.add_weight(name='weights_periodic',
                                                shape=(input_shape[-1], self.kernel_size),
                                                initializer='uniform',
                                                trainable=True)
        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(self.kernel_size,),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        linear = tf.matmul(x, self.weights_linear) + self.bias_linear
        periodic = tf.math.sin(tf.matmul(x, self.weights_periodic) + self.bias_periodic)
        return tf.concat([linear, periodic], axis=-1)

def build_transformer_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = Time2Vec()(inputs)
    x = layers.Dense(64)(x)

    # Transformer block
    attention_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    attention_output = layers.LayerNormalization()(attention_output + x)

    ffn_output = layers.Dense(64, activation='relu')(attention_output)
    ffn_output = layers.Dense(64)(ffn_output)
    x = layers.LayerNormalization()(ffn_output + attention_output)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    return model
