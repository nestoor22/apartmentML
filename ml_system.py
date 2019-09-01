import joblib
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder


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
        label_to_num_transformer = LabelEncoder()
        one_hot_transformer = OneHotEncoder()

        labels_number = label_to_num_transformer.fit_transform(dataset[column].values.reshape(-1, 1))
        labels_to_binary = one_hot_transformer.fit_transform(labels_number.reshape(-1, 1)).toarray()

        one_hot_dataset = pd.DataFrame(labels_to_binary, columns=[column+'_'+str(int(i))
                                                                  for i in range(labels_to_binary.shape[1])])

        information_about_transformers[column] = {'transformer-objects': {'OneHotTransformer': one_hot_transformer,
                                                                          'LabelTransformer': label_to_num_transformer}}

        dataset = pd.concat([dataset.drop(columns=column), one_hot_dataset], axis=1)

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

    with open('transformers_for_apartment_dataset', 'wb') as f:
        joblib.dump(information_about_transformers, f)

    return rescaled_dataset


def normalize_data(data, data_stats):
    return (data-data_stats['mean']) / data_stats['std']


dataset = rescale_data(original_dataset)


def build_nn_model(train_input):

    model = keras.Sequential([keras.layers.Dense(512, activation=tf.nn.sigmoid, input_shape=[len(train_input.keys())]),
                              keras.layers.Dense(128, activation=tf.nn.relu),
                              keras.layers.Dense(128, activation=tf.nn.tanh),
                              keras.layers.Dense(512, activation=tf.nn.relu),
                              keras.layers.Dense(1, activation=tf.nn.sigmoid)])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    return model


def train_model_for_price_prediction():
    x = dataset.drop(columns=['Cost'])
    y = dataset['Cost']
    train_input, test_input, train_output, test_output = train_test_split(x, y, train_size=0.8,
                                                                          test_size=0.2, random_state=0)

    train_stats = train_input.describe().transpose()
    normalize_train_data = normalize_data(train_input, train_stats)
    normalize_test_data = normalize_data(test_input, train_stats)

    model = build_nn_model(train_input)

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(normalize_train_data, train_output, epochs=1000, batch_size=16,
              validation_split=0.15, callbacks=[callback], verbose=1)

    test_predictions = model.predict(normalize_test_data).flatten()

    test_predictions = information_about_transformers['Cost']['transformer-object'].\
        inverse_transform(test_predictions.reshape(-1, 1))

    with open('models/price_prediction_model.json', 'w') as f:
        f.write(model.to_json())


def train_model_for_area_prediction():
    x = dataset.drop(columns=['Area'])
    y = dataset['Area']
    train_input, test_input, train_output, test_output = train_test_split(x, y, train_size=0.8,
                                                                          test_size=0.2, random_state=0)

    train_stats = train_input.describe().transpose()
    normalize_train_data = normalize_data(train_input, train_stats)
    normalize_test_data = normalize_data(test_input, train_stats)

    model = build_nn_model(train_input)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(normalize_train_data, train_output, epochs=1000, batch_size=8,
              validation_split=0.1, callbacks=[callback], verbose=1)

    test_predictions = model.predict(normalize_test_data).flatten()

    test_predictions = information_about_transformers['Area']['transformer-object'].\
        inverse_transform(test_predictions.reshape(-1, 1))

    print(test_predictions)

    with open('models/area_prediction_model.json', 'w') as f:
        f.write(model.to_json())


def train_model_for_distance_to_center():
    x = dataset.drop(columns=['DistanceToCenter'])
    y = dataset['DistanceToCenter']
    train_input, test_input, train_output, test_output = train_test_split(x, y, train_size=0.8,
                                                                          test_size=0.2, random_state=0)

    train_stats = train_input.describe().transpose()
    normalize_train_data = normalize_data(train_input, train_stats)
    normalize_test_data = normalize_data(test_input, train_stats)

    model = build_nn_model(train_input)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(normalize_train_data, train_output, epochs=1000, batch_size=8,
              validation_split=0.1, callbacks=[callback], verbose=1)

    test_predictions = model.predict(normalize_test_data).flatten()

    test_predictions = information_about_transformers['DistanceToCenter']['transformer-object'].\
        inverse_transform(test_predictions.reshape(-1, 1))

    print(test_predictions)

    with open('models/distance_to_center_prediction_model.json', 'w') as f:
        f.write(model.to_json())


def train_model_for_rooms_prediction():
    x = dataset.drop(columns=['Rooms'])

    num_classes = len(dataset['Rooms'].drop_duplicates())

    y_raw = original_dataset['Rooms'].values

    y_raw = y_raw - min(y_raw)

    y = to_categorical(y_raw, num_classes)

    train_input, test_input, train_output, test_output = train_test_split(x, y, train_size=0.8,
                                                                          test_size=0.2, random_state=0)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, input_shape=(len(train_input.columns),)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])

    model.fit(train_input, train_output,
              batch_size=8, epochs=1000, verbose=1)

    print(model.evaluate(test_input, test_output))

    with open('models/rooms_prediction_model.json', 'w') as f:
        f.write(model.to_json())

train_model_for_rooms_prediction()