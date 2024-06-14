import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, shuffle=True, id_to_color=None):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.id_to_color = id_to_color
        self.indexes = np.arange(len(image_paths))

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        if self.shuffle:
            np.random.shuffle(indexes)
        image_paths = [self.image_paths[i] for i in indexes]
        mask_paths = [self.mask_paths[i] for i in indexes]

        x = [tf.io.read_file(path) for path in image_paths]
        x = [tf.image.decode_image(img, channels=3, expand_animations=False) for img in x]
        x = tf.image.resize(x, (128, 128)) 
        x = x / 255.0

        y = [tf.io.read_file(path) for path in mask_paths]
        y = [tf.image.decode_image(img, channels=3, expand_animations=False) for img in y]
        y = tf.image.resize(y, (128, 128))
        y = [self.segmented_image_to_vector(img, self.id_to_color) for img in y]

        return np.stack(x), np.stack(y)
    
    
    def segmented_image_to_vector(self, segmented_image, id_to_color):
        mask = np.full( segmented_image.shape[:2], -1, dtype=int)
        closest_distances = np.full( segmented_image.shape[:2], np.inf)
        for id, color in id_to_color.items():
            distance =  np.linalg.norm(segmented_image - np.array(color).reshape(1, 1, -1), axis=-1)
            is_closer = closest_distances > distance
            mask = np.where( is_closer , id, mask)
            closest_distances = np.where( is_closer, distance, closest_distances)
        return tf.keras.utils.to_categorical(mask, len(id_to_color))

