import os

def run_spiders():
    os.system('scrapy crawl apartments_links_in_lviv -o lviv_pages_link.json && scrapy crawl get_apartment_info_in_lviv -o lviv_info.json')

