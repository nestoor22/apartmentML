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

original_dataset = pd.read_sql('SELECT * FROM apartment_info', sqlite3.connect('ApartmentsInfo.db'))\
    .drop(columns=['index'])
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
    columns_with_numeric_data, columns_with_str_data = search_different_types_column(dataset)

    dataset_with_numeric_columns = original_dataset[columns_with_numeric_data]
    dataset_with_str_columns = original_dataset[columns_with_str_data]

    scaled_numeric_columns = numeric_scaler.fit_transform(np.array(dataset_with_numeric_columns))
    scaled_string_columns = string_scaler.fit_transform(np.array(dataset_with_str_columns).ravel())
    scaled_string_columns = scaled_string_columns.reshape(len(dataset_with_str_columns),
                                                          len(columns_with_str_data))

    dataset_with_numeric_columns[columns_with_numeric_data] = scaled_numeric_columns
    dataset_with_str_columns[columns_with_str_data] = scaled_string_columns.reshape(len(dataset_with_str_columns),
                                                                                    len(columns_with_str_data))

    rescaled_dataset = pd.concat([dataset_with_numeric_columns, dataset_with_str_columns], axis=1)

    return rescaled_dataset


def split_data_and_train_model(dataset, train_s, test_s):
    rescaled_dataset = rescale_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(rescaled_dataset, rescaled_dataset['Cost'],
                                                        test_size=test_s, train_size=train_s)

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.columns.shape[0], activation=tf.nn.sigmoid,
                    kernel_initializer='normal'))
    model.add(Dense(256, activation=tf.nn.relu, kernel_initializer='normal'))
    model.add(Dense(256, activation=tf.nn.relu, kernel_initializer='normal'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation=tf.nn.relu, kernel_initializer='normal'))
    model.add(Dense(256, activation=tf.nn.relu, kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal', activation=keras.activations.linear))
    model.compile(loss='mean_squared_error', optimizer=RMSprop(0.0001), metrics=['mean_absolute_error',
                                                                                 'mean_squared_error'])
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=[early_stop])

    model_js = model.to_json()
    with open('ml_model_in_json', 'w') as json_file:
        json_file.write(model_js)

    predicted = model.predict(X_test).flatten()
    X_test['Cost'] = predicted

    X_test_string_decode = string_scaler.inverse_transform(X_test[['BuildingType', 'Condition', 'WallsMaterial']])
    X_test_string_df = pd.DataFrame(X_test_string_decode, columns=['BuildingType', 'Condition', 'WallsMaterial'])

    X_test_numeric = X_test.drop(columns=['BuildingType', 'Condition', 'WallsMaterial']).reset_index()\
        .drop(columns='index')
    X_test_numeric_decode = numeric_scaler.inverse_transform(X_test_numeric)
    X_test_numeric_df = pd.DataFrame(X_test_numeric_decode, columns=X_test_numeric.columns)

    res = pd.concat([X_test_numeric_df, X_test_string_df], axis=1)
    res.to_excel('result.xlsx')


def run_training_and_save_apartment_price_model():
    split_data_and_train_model(original_dataset, 0.8, 0.2)
