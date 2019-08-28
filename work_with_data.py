import sqlite3
import pandas as pd


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
    database_connect.cursor().execute("""DELETE FROM apartment_info WHERE CeilingHeight > 5""")
    database_connect.commit()


def fill_absent_data_and_remove_incorrect():

    for column_name in apartment_data_frame:
        for row in range(number_of_rows):
            if column_name == 'LivingArea' and pd.isna(apartment_data_frame['KitchenArea'][row]):
                apartment_data_frame[column_name][row] = float('{:.3f}'.format(apartment_data_frame['Area'][row] * 0.35))
                apartment_data_frame['KitchenArea'][row] = float('{:.3f}'.format(apartment_data_frame['Area'][row] * 0.2))

            elif column_name == 'LivingArea' and float(apartment_data_frame['KitchenArea'][row]):
                apartment_data_frame[column_name][row] = float('{:.3f}'.format((apartment_data_frame['Area'][row] -
                                                                                apartment_data_frame['KitchenArea'][row]) * 0.6))

            elif column_name == 'WallsMaterial' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 'unknown'

            elif column_name == 'BuildingType' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 'unknown'

            elif column_name == 'Condition' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 'unknown'

            elif column_name == 'Balconies' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 0.0

            elif column_name == 'CeilingHeight' and pd.isna(apartment_data_frame[column_name][row]) \
                    or apartment_data_frame[column_name][row] == 'h' or apartment_data_frame[column_name][row] == 'N':
                apartment_data_frame[column_name][row] = float(2.75)

            elif column_name == 'Floor' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 0.0

            elif column_name == 'Floors' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 0.0

            elif column_name == 'DistanceToCenter' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 0.0

    database_connect.execute('DROP TABLE apartment_info')
    database_connect.commit()
    apartment_data_frame.to_sql('apartment_info', database_connect)
    remove_low_cost()
    remove_too_big_area()
    remove_incorrect_ceiling_height()


def change_building_types():
    database_connect.cursor().execute("""UPDATE apartment_info SET BuildingType = 'House'
     WHERE BuildingType='Mansion' OR BuildingType='Cottage' OR BuildingType='House in a cottage town' 
     OR BuildingType='Townhouse' OR BuildingType='House in a cottage'""")

    database_connect.cursor().execute("""DELETE FROM apartment_info WHERE BuildingType='Part of the house'""")
    database_connect.commit()


change_building_types()