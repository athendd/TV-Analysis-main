import logging
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
    )

class JsonSerializer:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, character_name, data):
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', character_name)
        fp = self.out_dir / f'{safe_name}.json'
        with fp.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        logger.info('Saved %s', fp)

        return fp