import os
import re
import json
import gmaps
import googlemaps
import mtranslate

information_about_apartments_in_lviv = json.load(open('lviv_info.json'))

gmaps.configure(api_key='AIzaSyCs6bbRbEjmCbKnihNcL5jxQeZ0-D9mj4c')   # Your api here
api = googlemaps.Client(key='AIzaSyCs6bbRbEjmCbKnihNcL5jxQeZ0-D9mj4c')

translate_dict = {'Ціна': 'cost', 'Ціна $': 'cost $', 'Адреса': 'address', 'Кімнат': 'rooms', 'Поверх': 'floor',
                  'Житлова площа': 'living_area', 'Загальна площа': 'area', 'Площа кухні': 'kitchen_area',
                  'Поверховість': 'floors', 'Висота стелі': 'ceiling_height', 'Балконів': 'balconies',
                  'Матеріал стін': 'walls_material', 'Стан': 'conditions', 'Тип будівлі': 'building_type'}

cashed = {}
useless_keys = ['Площа ділянки', 'Днів на сайті', 'Код', 'Оновлено']


def refactor_string_from_file_with_scrapped_data():

    list_of_lists_with_info = []

    for i in range(len(information_about_apartments_in_lviv)):
        string_with_info = 'Адреса: '
        string_with_info = string_with_info + ''.join(information_about_apartments_in_lviv[i]['Address'])
        string_with_info += ''.join(information_about_apartments_in_lviv[i]['info'])

        changed_info_string = re.sub(r'\\n|\s\s+', '&', string_with_info)
        changed_info_string = re.sub('&&', '&', changed_info_string)

        structured_info = re.findall(r"[\w\s\d.,/$:]+", changed_info_string)
        list_of_lists_with_info.append(structured_info)

    return list_of_lists_with_info


def create_list_of_dicts():

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
    list_of_dicts = create_list_of_dicts()

    print('List of dict with scrapped data created')

    for info_dt in list_of_dicts:
        result_dict = {}
        keys = [key for key in info_dt.keys() if key not in useless_keys]

        for key in keys:
            if 'грн/м' in info_dt[key]:
                result_dict['cost'] = int(info_dt['Ціна $'].replace('$', '').replace(' ', '')) * \
                                      int(float(info_dt['Загальна площа']))

            elif key == 'Ціна':
                result_dict['cost'] = int(info_dt['Ціна $'].replace('$', '').replace(' ', ''))

            elif key == 'Ціна $':
                continue

            elif key == 'Адреса':
                result_dict[translate_dict[key]] = info_dt[key]

            elif key == 'Висота стелі':
                result_dict[translate_dict[key]] = info_dt[key].replace('h', '').replace('N', '')\
                    .replace(' ', '').replace('/', '').replace('m', '').replace('м', '')

            else:
                if info_dt[key] not in cashed and key in translate_dict:
                    try:
                        result_dict[translate_dict[key]] = mtranslate.translate(info_dt[key], 'en')
                        cashed[info_dt[key]] = result_dict[translate_dict[key]]

                    except Exception as error:
                        print(error)

                elif key in translate_dict:
                    result_dict[translate_dict[key]] = cashed[info_dt[key]]

                print('Translate one')

        json_for_db.append(result_dict)
        print('Appended to result')

    result = []

    for info_dt in json_for_db:

        for key in info_dt:
            if key == 'address':
                try:
                    info_dt['distance_to_center'] = float(api.directions(info_dt[key], 'Львів Оперний театр')[0]['legs']
                                                        [0]['distance']['text'].replace(' km', ''))
                    info_dt.pop(key)
                    print('Save distance')
                except Exception:
                    print('Distance error')
                    info_dt['distance_to_center'] = 0.0
                    info_dt.pop(key)
                    continue

        result.append(info_dt)

    with open('info_to_db.json', 'w') as json_file:
        json.dump(result, json_file, ensure_ascii=False)

    os.remove('info.json')
    os.remove('pages_link.json')
    return 1


create_json_for_db()
