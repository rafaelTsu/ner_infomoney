from scrapy.spiders import SitemapSpider

class InfomoneySitemapSpider(SitemapSpider):
    name = "infomoney_sitemap"
    allowed_domains = ["infomoney.com.br"]
    sitemap_urls = [
        'https://www.infomoney.com.br/news-sitemap.xml'
    ]

    def parse(self, response):
        self.log('Parsing URL: {}'.format(response.url))
        
        # Coleta conteúdo das notícias
        text_paragraphs = response.css('article.im-article p::text').getall()
        text = ' '.join(text_paragraphs)

        if text:
            yield {
                'url': response.url,
                'text': text.strip()
            }