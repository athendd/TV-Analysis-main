import scrapy
from bs4 import BeautifulSoup

class BlogSpider(scrapy.Spider):
    name = 'narutospider'
    start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu']

    def parse(self, response):
        #Go through pages with data on each move
        for href in response.css('.smw-columnlist-container')[0].css("a::attr(href)").extract():
            extracted_data = scrapy.Request("https://naruto.fandom.com"+href,
                           callback=self.parse_jutsu)
            yield extracted_data
            
        #Iterates through pages
        for next_page in response.css('a.mw-nextlink'):
            yield response.follow(next_page, self.parse)
    
    def parse_jutsu(self, response):
        jutsu_name = response.css("span.mw-page-title-main::text").extract()[0]
        
        #Removes unncessary spaces from the name
        jutsu_name = jutsu_name.strip()

        div_selector = response.css("div.mw-parser-output")[0]
        
        #Gets the html code   
        div_html = div_selector.extract()

        #Gets the first div in the html code
        soup = BeautifulSoup(div_html).find('div')

        jutsu_type=""
        if soup.find('aside'):
            
            #Extracts aside section of webpage
            aside = soup.find('aside')

            for cell in aside.find_all('div',{'class':'pi-data'}):
                if cell.find('h3'):
                    cell_name = cell.find('h3').text.strip()
                    if cell_name == "Classification":
                        jutsu_type = cell.find('div').text.strip()

        #Removes aside section
        soup.find('aside').decompose()

        jutsu_description = soup.text.strip()

        #Gets everything before trivia which is only the description
        jutsu_description = jutsu_description.split('Trivia')[0].strip()

        return dict (
            jutsu_name = jutsu_name,
            jutsu_type = jutsu_type,
            jutsu_description = jutsu_description
        )