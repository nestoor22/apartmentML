import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

original_dataset = pd.read_sql('SELECT * FROM apartment_info', sqlite3.connect('ApartmentsInfo.db'))
original_dataset["ceiling_height"] = pd.to_numeric(original_dataset['ceiling_height'])

apartment_data_frame = original_dataset


def plot_correlation():
    corr = apartment_data_frame.corr()
    f = plt.figure(figsize=(16, 15))
    plt.matshow(corr, fignum=f.number)
    plt.xticks(range(len(corr.columns)), corr.columns, fontsize=14, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix\n\n\n\n', fontsize=26, loc='center')
    plt.show()


def plot_area_histogram():
    _, ax = plt.subplots()
    ax.hist(apartment_data_frame['area'], color='#539caf')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('Histogram of Area')
    plt.show()


def plot_bar_corr_between_areas():
    _, ax = plt.subplots()
    ax.bar(apartment_data_frame['living_area'], apartment_data_frame['kitchen_area'], color='#539caf', align='center')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('')
    plt.show()


def plot_bar_corr_between_cost():
    _, ax = plt.subplots(figsize=(15,10))
    ax.bar(apartment_data_frame['distance_to_center'], apartment_data_frame['cost'], color='green', align='center')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('')
    plt.show()


def plot_build_type_and_cost():
    _, ax = plt.subplots()
    ax.scatter(apartment_data_frame['Cost'], apartment_data_frame['BuildingType'], s=3, color='red', alpha=1)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('Plot')
    plt.show()
