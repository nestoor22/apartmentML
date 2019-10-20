import os


def run_spiders():
    os.system('scrapy crawl apartments_links_in_lviv -o lviv_apartment_page_links.json &&'
              ' scrapy crawl get_apartment_info_in_lviv -o lviv_info.json')
    os.system('scrapy crawl apartments_links_in_kyiv -o kyiv_apartment_page_links.json &&'
              ' scrapy crawl get_apartment_info_in_kyiv -o kyiv_info.json')
