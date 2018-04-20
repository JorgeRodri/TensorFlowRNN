import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = dict()
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets():
    dataset = tf.keras.utils.get_file(
                    fname="aclImdb.tar.gz",
                    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                    extract=True)
    train = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    test = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))
    return train, test


# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

train_df, test_df = download_and_load_datasets()
print(train_df.head())

# Training input on the whole training set with no limit on training epochs.
# train_input_fn = tf.estimator.inputs.pandas_input_fn(
#     train_df, train_df["polarity"], num_epochs=None, shuffle=True)
#
# # Prediction on the whole training set.
# predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
#     train_df, train_df["polarity"], shuffle=False)
# # Prediction on the test set.
# predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
#     test_df, test_df["polarity"], shuffle=False)
#
# embedded_text_feature_column = hub.text_embedding_column(
#     key="sentence",
#     module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

