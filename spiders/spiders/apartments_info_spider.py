import scrapy
import json


class LvivInfoScrapper(scrapy.Spider):
    name = 'get_apartment_info_in_lviv'

    def start_requests(self):
        pages_link_in_json = json.load(open('lviv_pages_link.json'))

        pages_urls = ['https://www.real-estate.lviv.ua/sale-kvartira/Lviv/новобудови']
        for link in pages_link_in_json:
            pages_urls.append(link['link'])

        urls = pages_urls
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for link in response.css('div[class="col-sm-6 col-dense-right"]'):
            if link.css('a[class="object-address"]::attr(href)').get() is not None:
                apart_link = 'https://www.real-estate.lviv.ua' + link.css('a[class="object-address"]::attr(href)').get()
                yield response.follow(apart_link, callback=self.parse_info)

    def parse_info(self, response):
        for info in response.css('div[class="col-md-8"]'):
            yield {'info': info.css('li[class="col-sm-6 col-dense-left"]::text').getall(),
                   'Address': info.css('h1[class="thin row-dense row-dense-top"]::text').get()}


class KyivInfoScrapper(scrapy.Spider):
    name = 'get_apartment_info_in_kyiv'

    def start_requests(self):
        pages_link_in_json = json.loads(open('kyiv_pages_link.json', 'r'))
        pages_urls = []
        for link in pages_link_in_json:
            pages_urls.append(link['link'])

        for url in pages_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for info in response.css('div[class="object-overall"]'):
            yield {'info': info.css('div[class="value"]::text').getall()[5:]}