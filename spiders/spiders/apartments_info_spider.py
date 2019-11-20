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
        necessary_properties = ['К-сть кімнат/Розташування', 'Площа (загальна/житлова/кухні):',
                                'Поверх/К-сть поверхів:', 'Матеріал стін:', 'Ремонт (стан):']
        for info in response.css('div[class="object-overall"]'):
            info_dict = {'Address': info.css('div[id="object-address"] a::text').getall()}
            all_necessary_info = dict()
            all_necessary_info['rooms_info'] = info.css('div[id="object-rooms"] a::text').get() \
                if info.css('div[id="object-rooms"] a::text').get() else None

            all_necessary_info['area_info'] = info.css('div[id="object-squares"] div[class="value"]::text').get()

            all_necessary_info['floors_info'] = info.css('div[id="object-floors"] div[class="value"]::text').get()

            all_necessary_info['walls_material'] = info.css('div[id="object-materials"] div[class="value"]::text').get()

            all_necessary_info['conditions_info'] = info.css('div[id="object-levels"] div[class="value"]::text').get()
            if any(all_necessary_info.values()):
                info_dict.update({'info': all_necessary_info})
                yield all_necessary_info