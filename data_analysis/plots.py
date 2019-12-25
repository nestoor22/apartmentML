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


def plot_area_histogram():
    _, ax = plt.subplots()
    ax.hist(data['area'], color='#539caf')
    ax.set_ylabel('Count')
    ax.set_xlabel('Square meters')
    ax.set_title('Histogram of Area')
    plt.savefig('area_histogram.png')


def plot_bar_corr_between_areas():
    _, ax = plt.subplots()
    ax.bar(data['living_area'], data['kitchen_area'], color='#539caf', align='center')
    ax.set_ylabel('Living area')
    ax.set_xlabel('Kitchen area')
    ax.set_title('')
    plt.savefig('areas_bar.png')


def plot_bar_corr_between_cost():
    _, ax = plt.subplots(figsize=(15,10))
    ax.bar(data['distance_to_center'], data['cost'].apply(func=lambda x: x//1000), color='green', align='center')
    ax.set_ylabel('Price')
    ax.set_xlabel('Distance')
    ax.set_title('')
    plt.savefig('distance_price_bar.png')


def plot_build_type_and_cost():
    _, ax = plt.subplots()

    new_data = data.drop(data.loc[data['building_type'] == 'unknown'].index)

    ax.bar(new_data['building_type'],
           new_data['cost'].apply(func=lambda x: x//1000), color='red')

    ax.set_ylabel('Price (thousands)')
    ax.set_xlabel('Building type')
    ax.set_title('Building cost')
    plt.savefig('bulding_type_price_bar.png')


def plot_city_count():
    _, ax = plt.subplots()
    ax.hist(data['city'], color='green')
    ax.set_ylabel('Count of proposals')
    ax.set_xlabel('City name')
    ax.set_title('City-Buildings')
    plt.savefig('city_building_count_histogram.png')


def plot_city_price():
    _, ax = plt.subplots()
    ax.scatter(data['city'], data['cost'].apply(func=lambda x: x//1000), color='red')
    ax.set_ylabel('Price (thousands)')
    ax.set_xlabel('City')
    ax.set_title('Price distrubution')
    plt.savefig('city_price_scatter.png')


def plot_and_save_all():
    plot_city_price()
    plot_bar_corr_between_cost()
    plot_bar_corr_between_areas()
    plot_city_count()
    plot_area_histogram()
    plot_build_type_and_cost()
    plot_correlation()