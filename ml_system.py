import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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
        information_about_transformers[column] = {'transformer-object': [numeric_transformer]}
    return dataset


def scale_label_columns(dataset):

    for column in dataset:
        string_transformer = LabelEncoder()
        numeric_transformer = MinMaxScaler(feature_range=(0, 1))

        labels_to_numbers = string_transformer.fit_transform(dataset[column].values.reshape(-1, 1))
        numbers_to_range = numeric_transformer.fit_transform(labels_to_numbers.reshape(-1, 1))
        information_about_transformers[column] = {'transformer-object': [numeric_transformer, string_transformer]}
        dataset[column] = numbers_to_range

    return dataset


def decode_numeric(dataset, columns):
    for column in columns:
        dataset[column] = information_about_transformers[column]['transformer-object'][0].\
            inverse_transform(dataset[column].values.reshape(-1, 1))

    return dataset


def decode_labels(dataset, columns):
    for column in columns:
        number_of_label = information_about_transformers[column]['transformer-object'][0].\
            inverse_transform(dataset[column].values.reshape(-1, 1))

        label_string = information_about_transformers[column]['transformer-object'][1].\
            inverse_transform(number_of_label.reshape(len(number_of_label),).astype(int))

        dataset[column] = label_string

    return dataset


def rescale_data(dataset):

    columns_with_numeric_data, columns_with_str_data = search_different_types_column(dataset)

    dataset_with_label_columns = scale_label_columns(original_dataset[columns_with_str_data])
    dataset_with_numeric_columns = scale_numeric_columns(original_dataset[columns_with_numeric_data])

    rescaled_dataset = pd.concat([dataset_with_numeric_columns, dataset_with_label_columns], axis=1)

    # decode_numeric(rescaled_dataset, columns_with_numeric_data)
    # decode_labels(rescaled_dataset, columns_with_str_data)

    return rescaled_dataset


print(rescale_data(original_dataset))