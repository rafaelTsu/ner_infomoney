import csv
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from infomoney_scraper.spiders.infomoney_sitemap_spider import InfomoneySitemapSpider
from scrapy import signals
from scrapy.signalmanager import dispatcher

# Configura o arquivo CSV
csv_file = 'dataset/noticias.csv'

# Inicializa o arquivo CSV com os cabeçalhos
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['URL', 'text'])

# Função para processar os itens e salvar no CSV
def process_item(item):
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([item['url'], item['text']])

# Configura o processo Scrapy
process = CrawlerProcess(get_project_settings())

dispatcher.connect(process_item, signal=signals.item_scraped)

# Executa o spider
process.crawl(InfomoneySitemapSpider)
process.start()
