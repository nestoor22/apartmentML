import os
import re
import json
import mtranslate
from helpers import get_distance
from db_work.download_apartments_db import load_apartments_info_to_db

information_about_apartments = json.load(open('../json_files/kyiv_info.json'))

CASHED = {}


def create_json_for_db():

    result = []

    for info_dict in information_about_apartments:
        apartment_info = {}
        for key, value in info_dict.items():
            if key == 'cost':
                apartment_info[key] = value

            elif key == 'rooms_info':
                number_of_rooms = re.match(r'[\d]', value)
                if number_of_rooms:
                    apartment_info['rooms'] = re.match('[\d]', value).group()

            elif key == 'area_info':
                areas_info = value.split('/')
                areas_info_length = len(areas_info)
                if areas_info_length == 1:
                    apartment_info['area'] = areas_info[0]
                elif areas_info_length == 2:
                    apartment_info['area'], apartment_info['living_area'] = areas_info
                elif areas_info_length == 3:
                    apartment_info['area'], apartment_info['living_area'], apartment_info['kitchen_area'] = areas_info

                if 'area' in apartment_info:
                    apartment_info['square_meter_cost'] = info_dict['cost'] / apartment_info['area']

            elif key == 'floors_info':
                if value:
                    floors_info = value.split('/')
                    floors_info_length = len(floors_info)
                    if floors_info_length == 1:
                        apartment_info['floor'] = floors_info[0]
                    elif floors_info_length == 2:
                        apartment_info['floor'], apartment_info['floors'] = floors_info

            elif key in ['conditions', 'walls_material']:
                if value and value not in CASHED:
                    apartment_info[key] = mtranslate.translate(value, 'en')
                    CASHED[value] = apartment_info['conditions']

                elif value in CASHED:
                    apartment_info[key] = CASHED[value]

            elif key == 'address':
                apartment_info['distance_to_center'] = get_distance(value, 'Майдан Незалежності, Київ')

        apartment_info['building_type'] = 'New building'
        result.append(apartment_info)


create_json_for_db()