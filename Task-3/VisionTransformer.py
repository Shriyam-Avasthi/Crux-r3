import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, MultiHeadAttention, Resizing, Flatten, RandomRotation, RandomZoom, add


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        h, w, c = inputs.shape[1:]
        patches = tf.image.extract_patches( inputs, sizes = (1,self.patch_size, self.patch_size, 1), strides = (1,self.patch_size, self.patch_size,1), rates = (1,1,1,1), padding="VALID")
        vert_num_patches = h // self.patch_size
        horiz_num_patches = w // self.patch_size

        patches = tf.reshape(patches, ( batch_size, vert_num_patches * horiz_num_patches, self.patch_size**2 * c ) )

        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projected_patches = Dense( projection_dim )
        self.pos_embedding = Embedding( input_dim = num_patches, output_dim = projection_dim )

    def call(self, patch):
        positions = tf.expand_dims( tf.range( start = 0, limit = self.num_patches, delta = 1 ), axis = 0 )
        projected_patches = self.projected_patches(patch)
        encoded = projected_patches + self.pos_embedding(positions)
        return encoded
    
class MLP(tf.keras.layers.Layer):
    def __init__(self, input_units, output_units):
        super().__init__()
        self.dense_1 = Dense(input_units)
        self.dense_2 = Dense(output_units)
    
    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
    
class Encoder(tf.keras.layers.Layer):
    def __init__( self, num_heads, projection_dim, mlp_input_units, mlp_output_units ):
        super().__init__()
        self.layerNorm_1 = LayerNormalization()
        self.layerNorm_2 = LayerNormalization()
        self.mha = MultiHeadAttention( num_heads=num_heads, key_dim=projection_dim )
        self.mlp = MLP(mlp_input_units, mlp_output_units)
    def call(self, input):
        x = self.layerNorm_1(input)
        x = self.mha(x,x)
        attention = add([input,x])
        x = self.layerNorm_2(attention)
        x = self.mlp(x)
        x = add([attention, x])

        return x

class DataAugmentation(tf.keras.layers.Layer):
    def __init__(self, img_size):
        super().__init__()
        self.resizing = Resizing(img_size, img_size)
        self.rotation = RandomRotation(factor = (-0.05,0.05))
        self.flip = RandomZoom(height_factor=0.2, width_factor=0.2)
    
    def call(self, x):
        x = self.resizing(x)
        x = self.rotation(x)
        x = self.flip(x)
        return x

class ViT(tf.keras.Model):
    def __init__(self, img_size, num_layers, patch_size, num_patches, projection_dim, num_heads, transformer_mlp_input_units, transformer_mlp_output_units, mlp_input_units, mlp_output_units, num_classes):
        super().__init__()
        self.augmentation = DataAugmentation(img_size)
        self.patches = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)
        self.encoder_layers = []
        self.layerNorm = LayerNormalization()
        self.flatten = Flatten()
        self.mlp = MLP(mlp_input_units, mlp_output_units)
        self.out = Dense(num_classes)

        for i in range(num_layers):
            self.encoder_layers.append(Encoder( num_heads, projection_dim, transformer_mlp_input_units, transformer_mlp_output_units ))
        
    def call(self, x):
        x = self.augmentation(x)
        x = self.patches(x)
        x = self.patch_encoder(x)
        for i in range( len(self.encoder_layers) ):
            x = self.encoder_layers[i](x)
        x = self.layerNorm(x)
        x = self.flatten(x)
        x = self.mlp(x)
        x = self.out(x)
        return x