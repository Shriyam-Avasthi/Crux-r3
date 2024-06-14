import tensorflow as tf
from tensorflow.keras.layers import GlobalAvgPool2D, GlobalMaxPool2D, Activation, Dense, Conv2D, concatenate, multiply

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio = 16):
        super().__init__()
        self.averagePool = GlobalAvgPool2D()
        self.maxPool = GlobalMaxPool2D()
        self.mlp = tf.keras.Sequential([
            Dense( channels // reduction_ratio, activation = "relu" ),
            Dense( channels, activation = "relu" )
        ])
        self.sigmoid = Activation("sigmoid")
    
    def call(self, input):
        global_avg_pool = self.averagePool(input)
        global_max_pool = self.maxPool(input)
        avg_pool_features = self.mlp(global_avg_pool)
        max_pool_features = self.mlp(global_max_pool)
        x = avg_pool_features + max_pool_features
        x = self.sigmoid(x)
        return x
    
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv2D = Conv2D(1,kernel_size = (7,7), padding = "same" )
        self.sigmoid = Activation("sigmoid")
    
    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1)
        global_avg_pool = tf.expand_dims(avg_pool, axis=-1)

        max_pool = tf.reduce_max(x, axis=-1)
        global_max_pool = tf.expand_dims(max_pool, axis=-1)

        x = concatenate( [global_avg_pool, global_max_pool] )
        x = self.conv2D(x)
        x = self.sigmoid(x)
        return x
    
class CBAM( tf.keras.layers.Layer ):
    def __init__(self, channels, reduction_ratio = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
    def call(self, F):
        channel_attention = self.channel_attention(F)
        F_ = multiply([F,channel_attention])
        spatial_attention = self.spatial_attention(F_)
        F__ = multiply([F_, spatial_attention])
        return F__
         