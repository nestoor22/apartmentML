import sqlite3
import pandas as pd


database_connect = sqlite3.connect('ApartmentsInfo.db')
apartment_data_frame = pd.read_sql("""SELECT * FROM apartment_info""", database_connect)


number_of_rows = len(apartment_data_frame)


def remove_low_cost():
    database_connect.cursor().execute("""DELETE FROM apartment_info WHERE cost < 8000 OR cost is NULL""")
    database_connect.commit()


def remove_too_big_area():
    database_connect.cursor().execute("""DELETE FROM apartment_info WHERE  area > 2000 OR living_area < 10""")
    database_connect.commit()


def remove_incorrect_ceiling_height():
    database_connect.cursor().execute("""DELETE FROM apartment_info WHERE ceiling_height > 5""")
    database_connect.commit()


def fill_absent_data_and_remove_incorrect():

    for column_name in apartment_data_frame:
        for row in range(number_of_rows):
            if column_name == 'area':
                if apartment_data_frame['living_area'][row]:
                    apartment_data_frame['kitchen_area'][row] = float('{:.3f}'.format((apartment_data_frame['area'][row]
                                                                                      - apartment_data_frame['living_area'][row]) * 0.9))
                else:
                    apartment_data_frame['kitchen_area'][row] = float('{:.3f}'.format(apartment_data_frame['area'][row] * 0.2))

                if not apartment_data_frame['living_area'][row]:
                    apartment_data_frame['living_area'][row] = float('{:.3f}'.format(apartment_data_frame['area'][row] * 0.6))

                if not apartment_data_frame['kitchen_area'][row]:
                    apartment_data_frame['kitchen_area'][row] = float('{:.3f}'.format(apartment_data_frame['area'][row] * 0.2))

            elif column_name == 'walls_material' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 'unknown'

            elif column_name == 'building_type' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 'unknown'

            elif column_name == 'conditions' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 'unknown'

            elif column_name == 'balconies' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 0.0

            elif column_name == 'ceiling_height' and pd.isna(apartment_data_frame[column_name][row]) \
                    or apartment_data_frame[column_name][row] == 'h' or apartment_data_frame[column_name][row] == 'N':
                apartment_data_frame[column_name][row] = float(2.75)
                if apartment_data_frame[column_name][row] > 200:
                    apartment_data_frame[column_name][row] = apartment_data_frame[column_name][row] / 100

            elif column_name == 'floor' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 0.0

            elif column_name == 'floors' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 0.0

            elif column_name == 'distance_to_center' and pd.isna(apartment_data_frame[column_name][row]):
                apartment_data_frame[column_name][row] = 0.0

    database_connect.execute('DROP TABLE apartment_info')
    database_connect.commit()
    apartment_data_frame.to_sql('apartment_info', database_connect, index=False)
    remove_low_cost()
    remove_too_big_area()
    remove_incorrect_ceiling_height()


def change_building_types():
    database_connect.cursor().execute("""UPDATE apartment_info SET building_type = 'House'
     WHERE building_type='Mansion' OR building_type='Cottage' OR building_type='House in a cottage town' 
     OR building_type='Townhouse' OR building_type='House in a cottage'""")

    database_connect.cursor().execute("""DELETE FROM apartment_info WHERE building_type='Part of the house'""")
    database_connect.commit()
