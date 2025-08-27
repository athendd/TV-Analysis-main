import logging
from datetime import datetime, timezone
from quote_analyzer import QuoteAnalyzer
from page_fetcher import PageFetcher
from data_parser import HxHParser
from seralizer import JsonSerializer

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
    )

class CharacterDataService:
    def __init__(self, fetcher: PageFetcher, serializer: JsonSerializer, quote_analyzer: QuoteAnalyzer):
        self.fetcher = fetcher
        self.serializer = serializer 
        self.quote_analyzer = quote_analyzer 

    def build_character(self, character_name):
        url, soup = self.fetcher.fetch(character_name)
        parser = HxHParser(soup)
        data = parser.parse_all()

        data['Name'] = character_name.replace('_', ' ')
        data['_meta'] = {
            'source': url,
            'fetched_at': datetime.now(timezone.utc).isoformat(),
            'schema_version': 1,
        }

        quotes = data.get('Quotes') or []
        if quotes:
            logger.info('Analyzing quotes (%d) for %s', len(quotes), character_name)
            data['Quotes'] = self.quote_analyzer.process_quotes(quotes)
        else:
            logger.info('No quotes found for %s', character_name)

        return data

    def build_and_save(self, character_name):
        data = self.build_character(character_name)

        return self.serializer.save(character_name, data)


if __name__ == '__main__':
    svc = CharacterDataService()
    svc.build_and_save('Killua_Zoldyck')
