import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

original_dataset = pd.read_sql('SELECT * FROM apartment_info', sqlite3.connect('ApartmentsInfo.db'))
    # .drop(columns=['index'])

original_dataset["CeilingHeight"] = pd.to_numeric(original_dataset['CeilingHeight'])

information_about_transformers = {}


def search_different_types_column(dataset):
    numeric_columns = []
    string_columns = []
    for column in dataset:
        if dataset[column].dtype == np.int64 or dataset[column].dtype == np.float64:
            numeric_columns.append(column)

        elif dataset[column].dtype == object:
            string_columns.append(column)

    return numeric_columns, string_columns


def scale_numeric_columns(dataset):

    for column in dataset:
        numeric_transformer = MinMaxScaler(feature_range=(0, 1))
        dataset[column] = numeric_transformer.fit_transform(dataset[column].values.reshape(-1, 1))
        information_about_transformers[column] = {'transformer-object': numeric_transformer}

    return dataset


def scale_label_columns(dataset):

    for column in dataset:
        label_to_number_transformer = LabelEncoder()
        one_hot_transformer = OneHotEncoder()
        labels_number = label_to_number_transformer.fit_transform(dataset[column].values.reshape(-1, 1))
        labels_to_binary = one_hot_transformer.fit_transform(labels_number.reshape(-1, 1)).toarray()
        one_hot_dataset = pd.DataFrame(labels_to_binary, columns=[column+'_'+str(int(i))
                                                                  for i in range(labels_to_binary.shape[1])])

        information_about_transformers[column] = {'transformer-object': one_hot_transformer}
        dataset = pd.concat([dataset, one_hot_dataset], axis=1)

    return dataset


def decode_numeric(dataset, columns):
    for column in columns:
        dataset[column] = information_about_transformers[column]['transformer-object'].\
            inverse_transform(dataset[column].values.reshape(-1, 1))

    return dataset


def decode_labels(dataset, columns):
    for column in columns:
        binary_to_label = information_about_transformers[column]['transformer-object'].\
            inverse_transform(dataset[column].values.reshape(-1, 1))

        dataset[column] = binary_to_label

    return dataset


def rescale_data(dataset):

    columns_with_numeric_data, columns_with_str_data = search_different_types_column(dataset)

    dataset_with_label_columns = scale_label_columns(original_dataset[columns_with_str_data])
    dataset_with_numeric_columns = scale_numeric_columns(original_dataset[columns_with_numeric_data])

    rescaled_dataset = pd.concat([dataset_with_numeric_columns, dataset_with_label_columns], axis=1)

    return rescaled_dataset


# print(rescale_data(original_dataset))