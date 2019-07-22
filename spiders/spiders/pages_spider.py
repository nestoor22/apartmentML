import scrapy


class PageSpider(scrapy.Spider):
    name = "apartment"

    def start_requests(self):
        urls = [
            'https://www.real-estate.lviv.ua/sale-kvartira/Lviv/новобудови',
            'https://www.real-estate.lviv.ua/sale-house/Lviv'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for quote in response.css('li.page'):
            if quote.css('a[rel="next"]::attr(href)').get() is not None:
                link = 'https://www.real-estate.lviv.ua' + quote.css('a[rel="next"]::attr(href)').get()
                yield {'link': link}

        next_page = response.css('li.next a::attr(href)').get()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)