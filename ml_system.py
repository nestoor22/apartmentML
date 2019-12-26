import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from data_analysis.scalers import rescale_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from data_analysis.feature_selection import (select_features_for_regression_predictions,
                                             select_features_for_class_predictions,
                                             select_features_for_reg_prediction_with_trees,
                                             select_features_for_class_prediction_with_trees,
                                             remove_highly_correlated_features)


original_dataset = pd.read_sql('SELECT * FROM apartment_info', sqlite3.connect('db_work/ApartmentsInfo.db'))

original_dataset["ceiling_height"] = pd.to_numeric(original_dataset['ceiling_height'])
original_dataset = remove_highly_correlated_features(original_dataset)

_, information_about_transformers = rescale_data(original_dataset, one_hot_using=True)


def normalize_data(data, data_stats):
    return (data-data_stats['mean']) / data_stats['std']


def build_nn_model(train_input):

    model = keras.Sequential([keras.layers.Dense(512, activation=tf.nn.sigmoid, input_shape=[len(train_input.keys())]),
                              keras.layers.Dense(128, activation=tf.nn.relu),
                              keras.layers.Dense(128, activation=tf.nn.relu),
                              keras.layers.Dense(512, activation=tf.nn.relu),
                              keras.layers.Dense(1, activation=tf.nn.sigmoid)])

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['accuracy', 'mean_squared_error'])

    return model


def train_model_for_price_prediction():

    select_features = select_features_for_regression_predictions(original_dataset, what_predict='cost')

    new_data, _ = rescale_data(original_dataset[set(select_features+['cost', 'city'])], one_hot_using=True)

    x = new_data.drop(columns=['cost'])
    y = new_data['cost']

    train_input, test_input, train_output, test_output = train_test_split(x, y, train_size=0.8,
                                                                          test_size=0.2, random_state=0)

    model = build_nn_model(train_input)

    callback = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    model.fit(train_input, train_output, epochs=1000, batch_size=16,
              validation_split=0.15, callbacks=[callback])

    print(model.evaluate(test_input, test_output))

    with open('models/price_prediction_model.json', 'w') as f:
        f.write(model.to_json())
        model.save_weights('models/price_prediction_weights.h5')


def train_model_for_area_prediction():
    select_features = select_features_for_regression_predictions(original_dataset, what_predict='area')
    print(select_features)

    new_data, _ = rescale_data(original_dataset[set(select_features+['area', 'city'])], one_hot_using=True)

    x = new_data.drop(columns=['area'])
    y = new_data['area']
    train_input, test_input, train_output, test_output = train_test_split(x, y, train_size=0.8,
                                                                          test_size=0.2, random_state=0)

    model = build_nn_model(train_input)
    callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(train_input, train_output, epochs=1000, batch_size=8,
              validation_split=0.1, callbacks=[callback])

    print(model.evaluate(test_input, test_output))

    with open('models/area_prediction_model.json', 'w') as f:
        f.write(model.to_json())
        model.save_weights('models/area_prediction_weights.h5')


def train_model_for_distance_to_center():
    select_features = select_features_for_regression_predictions(original_dataset, what_predict='distance_to_center')
    print(select_features)

    new_data, _ = rescale_data(original_dataset[set(select_features+['distance_to_center', 'city'])],
                               one_hot_using=True)

    x = new_data.drop(columns=['distance_to_center'])
    y = new_data['distance_to_center']

    train_input, test_input, train_output, test_output = train_test_split(x, y, train_size=0.8,
                                                                          test_size=0.2, random_state=0)

    model = build_nn_model(train_input)

    callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(train_input, train_output, epochs=1000, batch_size=8,
              validation_split=0.1, callbacks=[callback])

    print(model.evaluate(test_input, test_output))

    with open('models/distance_to_center_prediction_model.json', 'w') as f:
        f.write(model.to_json())
        model.save_weights('models/distance_to_center_prediction_weights.h5')


def train_nn_model_for_rooms_prediction():
    select_features = select_features_for_class_predictions(original_dataset, what_predict='rooms')
    print(select_features)

    new_data, _ = rescale_data(original_dataset[set(select_features+['rooms', 'city'])], one_hot_using=True)

    x = new_data.drop(columns=['rooms'])
    x = x.values
    num_classes = len(new_data['rooms'].drop_duplicates())

    y_raw = original_dataset['rooms'].values

    y_raw = y_raw - min(y_raw)

    y = to_categorical(y_raw, num_classes)

    train_input, test_input, train_output, test_output = train_test_split(x, y, train_size=0.8,
                                                                          test_size=0.2, random_state=0)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, input_shape=(len(new_data.columns) - 1,)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(128, input_shape=(len(new_data.columns) - 1,)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    callback = keras.callbacks.EarlyStopping(monitor='acc', patience=10)
    model.fit(train_input, train_output,
              batch_size=16, epochs=1000, callbacks=[callback])

    print(model.evaluate(test_input, test_output))

    with open('models/rooms_prediction_model.json', 'w') as f:
        f.write(model.to_json())
        model.save_weights('models/rooms_prediction_weights.h5')


def train_decision_tree_model_for_rooms_prediction():
    select_features = select_features_for_class_predictions(original_dataset, what_predict='rooms')
    print(select_features)
    new_data, _ = rescale_data(original_dataset[set(select_features+['rooms', 'city'])])

    x = new_data.drop(columns=['rooms'])
    y = original_dataset['rooms']

    train_input, test_input, train_output, test_output = train_test_split(x, y, train_size=0.8,
                                                                          test_size=0.2, random_state=0)

    decision_tree_model = DecisionTreeClassifier(max_depth=100).fit(train_input, train_output)
    accuracy = decision_tree_model.score(test_input, test_output)
    print(accuracy)

    import pickle
    pkl_filename = "models/decision_tree_rooms_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(decision_tree_model, file)


def train_all_models():
    train_model_for_price_prediction()
    train_model_for_distance_to_center()
    train_model_for_area_prediction()
    train_nn_model_for_rooms_prediction()
    train_decision_tree_model_for_rooms_prediction()

    import joblib
    joblib.dump(information_about_transformers, 'models/transformers_info')
