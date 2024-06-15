import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, concatenate, MaxPool2D

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_size):
        super().__init__()
        self.conv2D = Conv2D( n_filters, kernel_size=kernel_size, padding="same")
        self.batchnorm = BatchNormalization()
        self.activation = Activation("relu")

    def call(self, x):
        x = self.conv2D(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x
    
class decoder_upsample_block(tf.keras.layers.Layer):
    def __init__(self, num_filters , kernel_size):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.convTranspose = Conv2DTranspose(self.num_filters, self.kernel_size, strides = 2, padding = 'same') 
        self.convBlock_1 =  ConvBlock(self.num_filters, self.kernel_size)
        self.convBlock_2 =  ConvBlock(self.num_filters, self.kernel_size)

    def call(self, input, skip_connection):
        x = self.convTranspose(input)
        x = concatenate( [ x , skip_connection ] )
        x = self.convBlock_1(x) 
        x = self.convBlock_2(x)
        return x
    
class encoder_downsample_block(tf.keras.layers.Layer):
    def __init__(self,num_filters, kernel_size):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.convBlock_1 = ConvBlock(self.num_filters, self.kernel_size)
        self.convBlock_2 = ConvBlock(self.num_filters, self.kernel_size)
        self.maxPool = MaxPool2D(pool_size = (2,2), strides = 2)
    def call(self,x):
        x = self.convBlock_1(x)
        x = self.convBlock_2(x)
        x = self.maxPool(x)
        return x