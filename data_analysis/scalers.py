import sqlite3
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder


def search_different_types_column(loaded_dataset):
    numeric_columns = []
    string_columns = []
    for column in loaded_dataset:
        if loaded_dataset[column].dtype == np.int64 or loaded_dataset[column].dtype == np.float64:
            numeric_columns.append(column)

        elif loaded_dataset[column].dtype == object:
            string_columns.append(column)

    return numeric_columns, string_columns


def scale_numeric_columns(loaded_dataset):
    information_about_transformers = {}
    for column in loaded_dataset:
        numeric_transformer = MinMaxScaler(feature_range=(0, 1))
        loaded_dataset[column] = numeric_transformer.fit_transform(loaded_dataset[column].values.reshape(-1, 1))
        information_about_transformers[column] = {'transformer-object': numeric_transformer}

    return loaded_dataset, information_about_transformers


def scale_label_columns(loaded_dataset, one_hot_using=False):
    information_about_transformers = {}
    for column in loaded_dataset:
        label_to_num_transformer = LabelEncoder()

        labels_number = label_to_num_transformer.fit_transform(loaded_dataset[column].values.reshape(-1, 1))

        if one_hot_using:
            one_hot_transformer = OneHotEncoder()
            labels_to_binary = one_hot_transformer.fit_transform(labels_number.reshape(-1, 1)).toarray()
            one_hot_dataset = pd.DataFrame(labels_to_binary, columns=[column+'_'+str(int(i))
                                                                      for i in range(labels_to_binary.shape[1])])
            loaded_dataset = pd.concat([loaded_dataset, one_hot_dataset], axis=1)

        information_about_transformers[column] = {'transformer-object': label_to_num_transformer}

        loaded_dataset[column] = labels_number

    return loaded_dataset, information_about_transformers


def decode_numeric(loaded_dataset, columns, information):
    for column in columns:
        loaded_dataset[column] = information[column]['transformer-object'].\
            inverse_transform(loaded_dataset[column].values.reshape(-1, 1))

    return loaded_dataset


def decode_labels(loaded_dataset, columns, information):
    for column in columns:
        binary_to_label = information[column]['transformer-object'].\
            inverse_transform(loaded_dataset[column].values.reshape(-1, 1))

        loaded_dataset[column] = binary_to_label

    return loaded_dataset


def rescale_data(loaded_dataset, one_hot_using=False):
    information_about_transformers = {}
    columns_with_numeric_data, columns_with_str_data = search_different_types_column(loaded_dataset)

    dataset_with_label_columns, info = scale_label_columns(loaded_dataset[columns_with_str_data],
                                                           one_hot_using=one_hot_using)

    information_about_transformers.update(info)

    dataset_with_numeric_columns, info = scale_numeric_columns(loaded_dataset[columns_with_numeric_data])

    information_about_transformers.update(info)
    rescaled_dataset = pd.concat([dataset_with_numeric_columns, dataset_with_label_columns], axis=1)

    return rescaled_dataset, information_about_transformers
