import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAvgPool2D, multiply

class SqueezeAndExcitationBlock(tf.keras.layers.Layer):
    def __init__(self, channels, ratio = 16):
        super().__init__()
        self.squeeze = GlobalAvgPool2D()
        self.excitation = tf.keras.Sequential(
            [
                Dense( channels//ratio, activation="relu" ),
                Dense( channels, activation="sigmoid" )
            ]
        )
    
    def call(self, input):
        x = self.squeeze(input)
        x = self.excitation(x)
        return multiply([input,x])
    
