import os


def run_spiders():
    if not os.path.exists('json_files'):
        os.mkdir('json_files')

    os.system('scrapy crawl apartments_links_in_lviv -o json_files/lviv_apartment_page_links.json &&'
              ' scrapy crawl get_apartment_info_in_lviv -o json_files/lviv_info.json')
    os.system('scrapy crawl apartments_links_in_kyiv -o json_files/kyiv_apartment_page_links.json &&'
              ' scrapy crawl get_apartment_info_in_kyiv -o json_files/kyiv_info.json')

run_spiders()
