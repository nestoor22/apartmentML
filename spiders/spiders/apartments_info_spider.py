import scrapy
import json


class LvivInfoScrapper(scrapy.Spider):
    name = 'get_apartment_info_in_lviv'

    def start_requests(self):
        pages_link_in_json = json.load(open('json_files/lviv_apartment_page_links.json'))

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
        pages_link_in_json = json.load(open('json_files/kyiv_apartment_page_links.json'))
        pages_urls = []
        for link in pages_link_in_json:
            pages_urls.append(link['link'])

        for url in pages_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for link in response.css('div[class="object-address"]'):
            if link.css('a::attr(href)').get() is not None:
                apart_link = 'https://100realty.ua' + link.css('a::attr(href)').get()
                yield scrapy.Request(url=apart_link, callback=self.parse_info)

    def parse_info(self, response):
        info_dict = {'cost': int(response.css('div[class="currency"] '
                                              'div[class="value"]::text').get().replace('$', '').replace(' ', ''))}

        for info in response.css('div[class="object-overall"]'):
            all_address_part = info.css('div[id="object-address"] a::text').getall()
            info_dict['address'] = f'{all_address_part[0]}, {all_address_part[1]}, {all_address_part[-1]}'

            info_dict['rooms_info'] = info.css('div[id="object-rooms"] a::text').get() \
                if info.css('div[id="object-rooms"] a::text').get() else None

            info_dict['area_info'] = info.css('div[id="object-squares"] div[class="value"]::text').get()

            info_dict['floors_info'] = info.css('div[id="object-floors"] div[class="value"]::text').get()

            info_dict['walls_material'] = info.css('div[id="object-materials"] div[class="value"]::text').get()

            info_dict['conditions'] = info.css('div[id="object-levels"] div[class="value"]::text').get()

            yield info_dict