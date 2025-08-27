import logging
from urllib.parse import quote
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
import requests
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
    )

BASE_URL = 'https://hunterxhunter.fandom.com/wiki/'

class PageFetcher:
    def __init__(self, base_url = BASE_URL):
        self.base_url = base_url
        self.session = self._make_session()

    def _slugify(self, name):
        slug_input = name.replace(' ', '_')

        return quote(slug_input, safe='_')

    def fetch(self, character_name):
        url = f'{self.base_url}{self._slugify(character_name)}'
        logger.info('Fetching %s', url)
        resp = self.session.get(url, timeout=20)
        if resp.status_code != 200:
            raise ValueError(f'HTTP {resp.status_code} for {url}')
        try:
            soup = BeautifulSoup(resp.content, 'lxml')
        except Exception:
            logger.warning('Using html.parser for %s because lxml parser failed', url)
            soup = BeautifulSoup(resp.content, 'html.parser')

        return url, soup

    @staticmethod
    def _make_session():
        s = requests.Session()
        s.headers.update({'User-Agent': 'HxH-CharBot/1.0 (+research; contact@example.com)'})
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        s.mount('https://', HTTPAdapter(max_retries=retries))
        
        return s