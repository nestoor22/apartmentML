import os


def run_spiders():
    os.system('scrapy crawl apartment_links -o pages_link.json && scrapy crawl get_apartment_info -o info.json')

