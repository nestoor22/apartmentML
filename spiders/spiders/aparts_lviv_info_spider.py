import scrapy
import json


class InfoScrapper(scrapy.Spider):
    name = 'get_apartment_info'

    def start_requests(self):
        pages_json = json.load(open('pages_link.json'))

        pages_urls = ['https://www.real-estate.lviv.ua/sale-kvartira/Lviv/новобудови']
        for link in pages_json:
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
        for quote in response.css('div[class="col-md-8"]'):
            yield {'info': quote.css('li[class="col-sm-6 col-dense-left"]::text').getall(),
                   'Address': quote.css('h1[class="thin row-dense row-dense-top"]::text').get()}