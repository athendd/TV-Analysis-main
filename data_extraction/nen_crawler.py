import scrapy
import re
import unicodedata

def clean_text(text, name):
    #Decode unicode escape sequences 
    try:
        text = bytes(text, 'utf-8').decode('unicode_escape')  
        text = text.encode('latin1').decode('utf-8')          
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass  

    #Remove content in parentheses, square brackets, and asterisks
    text = re.sub(r'\([^)]*\)', '', text)  
    text = re.sub(r'\[[^\]]*\]', '', text)  
    text = re.sub(r'\*+', '', text)         

    #Remove most Japanese characters
    text = re.sub(r'[\u3040-\u30ff\u4e00-\u9faf]', '', text)

    #Normalize and strip extra whitespace
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if name:
        text = text.split(',')[0]
    
    text = text.replace('/', ' ')
    text = re.sub(r"\\u[0-9a-fA-F]{4}", "", text) 
    text = text.replace('\"', '')

    return text.strip()

class HunterxHunterBlogSpider(scrapy.Spider):
    name = 'hunterxhunterspider'
    start_urls = ['https://hunterxhunter.fandom.com/wiki/Nen']

    def parse(self, response):
        base_url = 'https://hunterxhunter.fandom.com' 
        
        nen_user_links = response.css('div[style*="column-count"] ul li a::attr(href)').getall()
                        
        for link_suffix in nen_user_links:
            if link_suffix.startswith('/wiki/') and '#' not in link_suffix and ':' not in link_suffix:
                yield scrapy.Request(base_url + link_suffix, callback=self.parse_nen)
        
    def parse_nen(self, response):
        character_name_raw = response.url.split('/')[-1]
        character_name = character_name_raw.replace('_', ' ').strip()

        ability_table = response.css('table.abilitytable')

        if ability_table:
            current_ability_data = {}
            
            # Skip the first row by using :nth-child(n+2) in the selector
            rows = ability_table.css('tr:not(.abilitytable-header):not(.mobile-only):not(.hidden):nth-child(n+2)')

            for row in rows:
                th_elements = row.css('th')
                td_elements = row.css('td')

                type_th_text = th_elements.css('::text').getall()
                if type_th_text and "Type:" in "".join(type_th_text):
                    if current_ability_data:
                        current_ability_data["Character_Name"] = character_name
                        # Validate before yielding
                        if self.is_complete(current_ability_data):
                            yield current_ability_data
                        
                    current_ability_data = {
                        "Name": None,
                        "Types": [],
                        "Description": None
                    }

                    types_raw = " ".join(th_elements[0].css('::text').getall()).replace("Type:", "").strip()
                    current_ability_data["Types"] = [t.strip() for t in types_raw.split(',') if t.strip()]

                    name_parts = row.xpath('.//th[2]//text()[not(ancestor::span[contains(@class, "t_nihongo_kanji") or contains(@class, "t_nihongo_comma") or contains(@class, "reference") or contains(@class, "explain")]) or self::text()]').getall()
                    
                    cleaned_name = " ".join(name_parts).strip()
                    current_ability_data["Name"] = clean_text(cleaned_name, True) if cleaned_name else None

                elif td_elements and current_ability_data:
                    description_parts = td_elements[-1].css('::text').getall()
                    description_raw = "".join(description_parts).strip()
                    current_ability_data["Description"] = clean_text(description_raw, False)

            if current_ability_data:
                current_ability_data["Character_Name"] = character_name
                # Validate the last collected item before yielding
                if self.is_complete(current_ability_data):
                    yield current_ability_data

    def is_complete(self, data):
        for key, value in data.items():
            if value is None:
                return False
            if isinstance(value, str) and not value.strip():
                return False
            if isinstance(value, list) and not value:
                return False
        return True