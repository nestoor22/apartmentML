import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers import Dense

original_dataset = pd.read_sql('SELECT * FROM apartment_info', sqlite3.connect('ApartmentsInfo.db')).drop(columns=['index'])
numeric_scaler = MinMaxScaler(feature_range=(0, 1))
string_scaler = LabelEncoder()


def search_different_types_column(dataset):
    numeric_columns = []
    string_columns = []
    for column in dataset:
        if dataset[column].dtype == np.int64 or dataset[column].dtype == np.float64:
            numeric_columns.append(column)

        elif dataset[column].dtype == object:
            string_columns.append(column)

    return numeric_columns, string_columns


def rescale_data(dataset):
    columns_with_numeric_data, columns_with_string_data = search_different_types_column(dataset)

    dataset_with_numeric_columns = original_dataset[columns_with_numeric_data]
    dataset_with_string_columns = original_dataset[columns_with_string_data]

    scaled_numeric_columns = numeric_scaler.fit_transform(np.array(dataset_with_numeric_columns))
    scaled_string_columns = string_scaler.fit_transform(np.array(dataset_with_string_columns).ravel())
    scaled_string_columns = scaled_string_columns.reshape(len(dataset_with_string_columns), len(columns_with_string_data))

    dataset_with_numeric_columns[columns_with_numeric_data] = scaled_numeric_columns
    dataset_with_string_columns[columns_with_string_data] = scaled_string_columns.reshape(len(dataset_with_string_columns),
                                                                                          len(columns_with_string_data))

    rescaled_dataset = pd.concat([dataset_with_numeric_columns, dataset_with_string_columns], axis=1)

    return rescaled_dataset


def split_data(dataset, train_s, test_s):
    rescaled_dataset = rescale_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(rescaled_dataset, rescaled_dataset['Cost'],
                                                        test_size=test_s, train_size=train_s)
    return X_train, X_test, y_train, y_test


def create_and_train_ml_model():
    pass


split_data(original_dataset, 0.8, 0.2)