import os
import re
import json
import mtranslate
from helpers import get_distance
from db_work.download_apartments_db import load_apartments_info_to_db

information_about_apartments = json.load(open('../json_files/lviv_info.json'))

TRANSLATE_DICT = {'Ціна': 'cost', 'Ціна $': 'cost $', 'Адреса': 'address', 'Кімнат': 'rooms', 'Поверх': 'floor',
                  'Житлова площа': 'living_area', 'Загальна площа': 'area', 'Площа кухні': 'kitchen_area',
                  'Поверховість': 'floors', 'Висота стелі': 'ceiling_height', 'Балконів': 'balconies',
                  'Матеріал стін': 'walls_material', 'Стан': 'conditions', 'Тип будівлі': 'building_type'}

CASHED = {}
USELESS_KEYS = ['Площа ділянки', 'Днів на сайті', 'Код', 'Оновлено']


def refactor_string_from_file_with_scrapped_data():

    list_of_lists_with_info = []

    for i in range(len(information_about_apartments)):
        string_with_info = 'Адреса: '
        string_with_info = string_with_info + ''.join(information_about_apartments[i]['Address'])
        string_with_info += ''.join(information_about_apartments[i]['info'])

        changed_info_string = re.sub(r'\\n|\s\s+', '&', string_with_info)
        changed_info_string = re.sub('&&', '&', changed_info_string)

        structured_info = re.findall(r"[\w\s\d.,/$:]+", changed_info_string)
        list_of_lists_with_info.append(structured_info)

    return list_of_lists_with_info


def create_list_with_apartments_information():

    list_of_lists_with_info = refactor_string_from_file_with_scrapped_data()
    result = []

    for info_list in list_of_lists_with_info:
        result_dict = {}

        for i in info_list:
            key = re.findall(r"[\w\s$]+", i)
            value = re.findall(r"\s[\d]+[.]?[\s]?[\d]*[\s]?[\d]*[\sгрн]*[/м]*[$]?", i)

            if len(key) > 0 and key[0] == 'address':
                value = re.findall(r":[\w\s.,\d]+", i)
                result_dict[key[0]] = value[0].rstrip().replace(': ', '')

            elif len(key) != 0 and len(value) != 0:
                result_dict[key[0]] = value[0].rstrip().replace(': ', '')

            elif len(key) != 0 and len(value) == 0:
                value = re.findall(r":[\w\s.,\d]+", i)
                if len(value) > 0:
                    result_dict[key[0]] = value[0].rstrip().replace(': ', '')

        result.append(result_dict)
    return result


def create_json_for_db():
    json_for_db = []
    list_of_dicts = create_list_with_apartments_information()

    print('List of dict with apartments data created')

    for info_dt in list_of_dicts:
        result_dict = {}
        keys = [key for key in info_dt.keys() if key not in USELESS_KEYS]

        for key in keys:
            if 'грн/м' in info_dt[key]:
                result_dict['cost'] = int(info_dt['Ціна $'].replace('$', '').replace(' ', '')) * \
                                      int(float(info_dt['Загальна площа']))

            elif key == 'Ціна':
                result_dict['cost'] = int(info_dt['Ціна $'].replace('$', '').replace(' ', ''))

            elif key == 'Ціна $':
                continue

            elif key == 'Адреса':
                result_dict[TRANSLATE_DICT[key]] = info_dt[key]

            elif key == 'Висота стелі':
                result_dict[TRANSLATE_DICT[key]] = info_dt[key].replace('h', '').replace('N', '')\
                    .replace(' ', '').replace('/', '').replace('m', '').replace('м', '')

            else:
                if info_dt[key] not in CASHED and key in TRANSLATE_DICT:
                    try:
                        result_dict[TRANSLATE_DICT[key]] = mtranslate.translate(info_dt[key], 'en')
                        CASHED[info_dt[key]] = result_dict[TRANSLATE_DICT[key]]

                    except Exception as error:
                        print(error)

                elif key in TRANSLATE_DICT:
                    result_dict[TRANSLATE_DICT[key]] = CASHED[info_dt[key]]

                print('Translate one')

        json_for_db.append(result_dict)
        print('Appended to result')

    result = []

    for info_dt in json_for_db:

        for key in info_dt:
            if key == 'address':
                info_dt['distance_to_center'] = get_distance(info_dt[key], 'Львів Оперний театр')
                info_dt.pop(key)

        result.append(info_dt)

    load_apartments_info_to_db(data_to_db=result)

    os.remove('lviv_info.json')
    os.remove('lviv_apartment_page_links.json')
    return 1


create_json_for_db()
