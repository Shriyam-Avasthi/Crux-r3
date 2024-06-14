import pandas as pd
from sklearn.model_selection import train_test_split
import ast
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import TextVectorization

class Dataset():
    def __init__(self,dataset_path, max_length, batch_size, buffer_size):
        df = pd.read_csv(dataset_path)
        df = df[["Sentence", "Tag"]]
        self.df_train, self.df_test = train_test_split(df, test_size=0.2)
        self.le = LabelEncoder()

        self.vectorizer = TextVectorization( output_sequence_length=max_length, standardize="lower", output_mode="int")
        self.vectorizer.adapt( self.df_train["Sentence"] )

        self.max_length = max_length
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size

    def get_unique_tags(self, df):
        unique_tags = set( [tag for sub_tag in list(df.Tag.values) for tag in ast.literal_eval(sub_tag)] )
        unique_tags.add("[START]")
        unique_tags.add("[END]")
        unique_tags.add("[PAD]")
        return unique_tags

    def preprocess_data(self,df, max_length):
        
        unique_tags = self.get_unique_tags(df)
        self.le.fit(list(unique_tags))

        encoded_tags = []
        for tags in df["Tag"].to_list():
            lst = ast.literal_eval(tags)
            lst.insert(0, "[START]")
            lst.append("[END]")
            encoded_tags.append(self.le.transform(lst))
        
        preprocessed_tags = pad_sequences(encoded_tags,maxlen=max_length + 1, padding="post", value = 18)

        return preprocessed_tags

    def prepare_batch(self, sentences, tags):
        return ( self.vectorizer(sentences), tags[:,:-1] ), tags[:,1:] 

    def make_batches(self, ds):
        return (
            ds
            .shuffle(self.BUFFER_SIZE)
            .batch(self.BATCH_SIZE)
            .map(self.prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

    def get_data_generators(self):
        
        train_generator = tf.data.Dataset.from_tensor_slices( (self.df_train["Sentence"].values, self.preprocess_data(self.df_train, self.max_length) ))

        test_generator = tf.data.Dataset.from_tensor_slices( (self.df_test["Sentence"].values, self.preprocess_data(self.df_test, self.max_length)))

        train_batches = self.make_batches(train_generator)
        test_batches = self.make_batches(test_generator)

        return train_batches, test_batches

    def get_vocab_size(self):
        return( self.vectorizer.vocabulary_size() )
    
    def decode(self, vector):
        vocab = self.vectorizer.get_vocabulary()
        return( " ".join([vocab[each] for each in tf.squeeze(vector)]) )
    
    def vectorize(self, sentences):
        return self.vectorizer(sentences)
    
    def encode_label(self, lst):
        return self.le.transform(lst)
    
    def inverse_transform_label(self, lst):
        return self.le.inverse_transform(lst)