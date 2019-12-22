import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_sql('SELECT * FROM apartment_info', sqlite3.connect('../db_work/ApartmentsInfo.db'))


def plot_correlation():
    corr = data.corr()
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
    ax.hist(data['area'], color='#539caf')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('Histogram of Area')
    plt.show()


def plot_bar_corr_between_areas():
    _, ax = plt.subplots()
    ax.bar(data['living_area'], data['kitchen_area'], color='#539caf', align='center')
    ax.set_ylabel('Living area')
    ax.set_xlabel('Kitchen area')
    ax.set_title('')
    plt.show()


def plot_bar_corr_between_cost():
    _, ax = plt.subplots(figsize=(15,10))
    ax.bar(data['distance_to_center'], data['cost'], color='green', align='center')
    ax.set_ylabel('Price')
    ax.set_xlabel('Distance')
    ax.set_title('')
    plt.show()


def plot_build_type_and_cost():
    _, ax = plt.subplots()
    ax.bar(data['building_type'], data['cost'], color='red')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('Plot')
    plt.show()
