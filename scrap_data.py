import os


def run_spiders():
    os.system('scrapy crawl apartment -o pages_link.json && scrapy crawl get_apartment_info -o info.json')

