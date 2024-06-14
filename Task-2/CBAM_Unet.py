import tensorflow as tf
from CBAM import CBAM
from Unet import encoder_downsample_block, decoder_upsample_block, ConvBlock

class CBAMUnet( tf.keras.Model ):
    def __init__(self):
        super().__init__()
        self.encoderBlock_1 = encoder_downsample_block(num_filters=64, kernel_size=(5,5))
        self.down_CBAM_1 = CBAM(channels=64)
        self.encoderBlock_2 = encoder_downsample_block(num_filters=128, kernel_size=(5,5))
        self.down_CBAM_2 = CBAM(channels=128)
        self.encoderBlock_3 = encoder_downsample_block(num_filters=256, kernel_size=(3,3))
        self.down_CBAM_3 = CBAM(channels=256)
        self.bottleneck = ConvBlock(512,(3,3))
        self.SEbottleneck = CBAM(channels=512)
        self.decoderBlock_1 = decoder_upsample_block(num_filters=256, kernel_size=(3,3))
        self.up_CBAM_1 = CBAM( channels=256)
        self.decoderBlock_2 = decoder_upsample_block(num_filters=128, kernel_size=(5,5))
        self.up_CBAM_2 = CBAM(channels=128)
        self.decoderBlock_3 = decoder_upsample_block(num_filters=64, kernel_size=(5,5))
        self.up_CBAM_3 = CBAM(channels=64)
        self.final_conv_block = ConvBlock(35, (5,5))
    
    @tf.function
    def call(self, inputs):
        # print("UP "+ str(inputs.shape))
        down_1 = self.encoderBlock_1(inputs)
        down_SE_1 = self.down_CBAM_1(down_1)
        down_2 = self.encoderBlock_2(down_SE_1)
        down_SE_2 = self.down_CBAM_2(down_2)
        down_3 = self.encoderBlock_3(down_SE_2)
        down_SE_3 = self.down_CBAM_3(down_3)
        bottleneck = self.bottleneck(down_SE_3)
        SEbottleneck = self.SEbottleneck(bottleneck)
        up_1 = self.decoderBlock_1(SEbottleneck, down_SE_2)
        up_SE_1 = self.up_CBAM_1(up_1)
        up_2 = self.decoderBlock_2(up_SE_1, down_SE_1)
        up_SE_2 = self.up_CBAM_2(up_2)
        up_3 = self.decoderBlock_3(up_SE_2, inputs)
        up_SE_3 = self.up_CBAM_3(up_3)
        final_convBlock = self.final_conv_block(up_SE_3)
        # print(out.shape)
        return final_convBlock