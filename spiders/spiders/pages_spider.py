import scrapy


class LvivPageSpider(scrapy.Spider):
    name = "apartments_links_in_lviv"

    def start_requests(self):
        urls = [
            'https://www.real-estate.lviv.ua/sale-kvartira/Lviv/новобудови',
            'https://www.real-estate.lviv.ua/sale-house/Lviv'

        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for page_link in response.css('li.page'):
            if page_link.css('a[rel="next"]::attr(href)').get() is not None:
                link = 'https://www.real-estate.lviv.ua' + page_link.css('a[rel="next"]::attr(href)').get()
                yield {'link': link}

        next_page = response.css('li.next a::attr(href)').get()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)


class KyivPageScrapper(scrapy.Spider):
    name = 'apartments_links_in_kyiv'

    def start_requests(self):
        urls = ['https://100realty.ua/uk/realty_search/apartment/sale/cur_3']

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        i = 0
        for page_link in response.css('li[class="pager-item pager-next"]'):
            if page_link.css('a[rel="next"]::attr(href)').get() is not None and not i:
                i = 1
                yield {'link': page_link.css('a[rel="next"]::attr(href)').get()}
        next_page = response.css('li[class="pager-item pager-next"] a::attr(href)').get()
        if next_page is not None:
            yield scrapy.Request(url=next_page, callback=self.parse)