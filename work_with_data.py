import sqlite3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


matplotlib.pyplot.style.use('ggplot')
database_connect = sqlite3.connect('ApartmentsInfo.db')
apartment_data_frame = pd.read_sql("""SELECT * FROM apartment_info""", database_connect)

number_of_rows = len(apartment_data_frame)


def remove_low_cost():
    database_connect.cursor().execute("""DELETE FROM apartment_info WHERE Cost < 8000 OR Cost is NULL""")
    database_connect.commit()


def remove_too_big_area():
    database_connect.cursor().execute("""DELETE FROM apartment_info WHERE  Area > 2000 OR LivingArea < 10""")
    database_connect.commit()


def remove_incorrect_ceiling_height():
    database_connect.cursor().execute("""DELETE FROM apartment_info WHERE  CeilingHeight > 5""")
    database_connect.commit()


def fill_all_absent_data():

    for column_name in apartment_data_frame:
        for row in range(number_of_rows):
            if column_name == 'LivingArea' and pd.isna(apartment_data_frame['KitchenArea'][row]):
                apartment_data_frame[column_name][row] = float('{:.3f}'.format(apartment_data_frame['Area'][row] * 0.35))
                apartment_data_frame['KitchenArea'][row] = float('{:.3f}'.format(apartment_data_frame['Area'][row] * 0.2))

            elif column_name == 'LivingArea' and float(apartment_data_frame['KitchenArea'][row]):
                apartment_data_frame[column_name][row] = float('{:.3f}'.format((apartment_data_frame['Area'][row] -
                                                                                apartment_data_frame['KitchenArea'][row]) * 0.6))

            elif column_name == 'SquareMeterCost' and pd.isna(apartment_data_frame[column_name][row]) \
                                                  and not pd.isna(apartment_data_frame['Cost'][row]):

                apartment_data_frame[column_name][row] = apartment_data_frame['Cost'][row] / apartment_data_frame['LivingArea'][row]

            elif column_name == 'WallsMaterial' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 'unknown'

            elif column_name == 'BuildingType' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 'unknown'

            elif column_name == 'Condition' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 'unknown'

            elif column_name == 'Balconies' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 0

            elif column_name == 'CeilingHeight' and pd.isna(apartment_data_frame[column_name][row]) \
                    or apartment_data_frame[column_name][row] == 'h' or apartment_data_frame[column_name][row] == 'N':
                apartment_data_frame[column_name][row] = 2.75

            elif column_name == 'Floor' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = '0'

            elif column_name == 'Floors' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = '0'

            elif column_name == 'DistanceToCenter' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 0.0

    database_connect.execute('DROP TABLE apartment_info')
    database_connect.commit()
    apartment_data_frame.to_sql('apartment_info', database_connect)
    remove_low_cost()
    remove_too_big_area()
    remove_incorrect_ceiling_height()


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
    ax.hist(apartment_data_frame['Area'], color='#539caf')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('Histogram of Area')
    plt.show()


def plot_bar_corr_between_areas():
    _, ax = plt.subplots()
    ax.bar(apartment_data_frame['LivingArea'], apartment_data_frame['KitchenArea'], color='#539caf', align='center')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('')
    plt.show()


def plot_bar_corr_between_cost():
    _, ax = plt.subplots(figsize=(15,10))
    ax.bar(apartment_data_frame['DistanceToCenter'], apartment_data_frame['Cost'], color='green', align='center')
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


if __name__ == '__main__':
    # print(pd.isna(apartment_data_frame[['Cost','LivingArea']][:2]))
    # fill_all_absent_data()
    plot_build_type_and_cost()