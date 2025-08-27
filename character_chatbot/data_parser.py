import logging
import re
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
    )

class HxHParser:
    def __init__(self, soup):
        self.soup = soup

    def parse_all(self):
        data = {'Physical Features': {}}
        self._parse_aside(into=data)
        self._parse_sections(into=data)

        return data

    def _parse_aside(self, into):
        aside = self.soup.find('aside')
        if aside is None:
            logger.warning('No infobox <aside> found')
            return

        physical_features = ['hair', 'eyes', 'height', 'weight', 'blood', 'gender']
        for feature in physical_features:
            into['Physical Features'][feature.capitalize()] = self._find_data_source(aside, feature)

        other_traits = ['affiliation', 'previous affiliation', 'status', 'birthday', 'age', 'type', 'relatives']
        for trait in other_traits:
            if trait == 'age':
                into[trait.capitalize()] = self._get_age(self._find_data_source(aside, trait))
            else:
                into[trait.capitalize()] = self._find_data_source(aside, trait)

    def _find_data_source(self, aside_object, data_source_name):
        element = aside_object.find(attrs={'data-source': data_source_name})
        if element is None:
            return None

        if data_source_name == 'relatives':
            container = aside_object.find('div', {'data-source': 'relatives'})
            if not container:
                return {}
            val = container.find(class_='pi-data-value')
            if not val:
                return {}

            relatives = {}
            for a in val.find_all('a'):
                name = a.get_text(strip=True)
                bits = []
                for sib in a.next_siblings:
                    if getattr(sib, 'name', None) == 'br':
                        break
                    bits.append(sib.get_text(' ', strip=True) if hasattr(sib, 'get_text') else str(sib))
                tail = ' '.join(bits).strip()
                m = re.search(r'\(([^)]+)\)', tail)
                relationship = (m.group(1).strip().replace('"', "") if m else tail).strip()
                if not relationship:
                    continue
                low = relationship.lower()
                if 'status unknown' in low or 'unknown relation' in low or name.lower().startswith('unnamed'):
                    continue
                relatives[name] = relationship

            return relatives

        if data_source_name in ['affiliation', 'previous affiliation']:
            vals = []
            for a_ele in element.find_all('a'):
                text = self._clean_text(a_ele.get_text(), strip_parens=False)
                if text:
                    vals.append(text)
            return ', '.join(vals)

        text = element.get_text()
        words = text.split()
        if data_source_name == 'type':
            print(words)
        start_idx = 2 if data_source_name in ['blood', 'eyes', 'hair'] else 1
        remaining_words = words[start_idx:] if len(words) > start_idx else []
        new_word = ' '.join(remaining_words)
        if data_source_name in ['birthday', 'height', 'weight', 'hair', 'eyes']:
            new_word = new_word.split('(')[0]

        element_text = new_word.replace('\n', '')

        return self._clean_text(element_text)

    def _parse_sections(self, into):
        sections = [
            'Appearance',
            'Personality',
            'Background',
            'Plot',
            'Equipment',
            'Abilities & Powers',
            'Quotes',
        ]

        for sec in sections:
            if sec == 'Plot':
                into.setdefault('Plot', {})
            elif sec == 'Equipment':
                into.setdefault('Equipment', {})
            elif sec == 'Abilities & Powers':
                into.setdefault('Abilities & Powers',
                                {'Description': '', 'Nen': {'Description': ''}})
            elif sec == 'Quotes':
                into.setdefault('Quotes', [])
            else:
                into.setdefault(sec, '')

            self._parse_section(sec, into)

    def _find_section_h2(self, title):
        span = self.soup.select_one(f'h2 span.mw-headline[id="{title}"]')
        if span:
            h2 = span.find_parent('h2')
            if h2:
                return h2

        for sp in self.soup.select('h2 span.mw-headline'):
            if sp.get_text(strip=True).lower() == title.lower():
                h2 = sp.find_parent('h2')
                if h2:
                    return h2

        def canon(s):
            s = (s or '').lower().replace('&', 'and')

            return re.sub(r'[^a-z0-9]+', '', s)

        target = canon(title)
        for h2 in self.soup.find_all('h2'):
            if canon(h2.get_text()) == target:
                return h2
        for h2 in self.soup.find_all('h2'):
            if target in canon(h2.get_text()):
                return h2

        return None

    def _parse_section(self, section_title, into):
        header = self._find_section_h2(section_title)
        if not header:
            logger.warning("Section header not found: %s", section_title)
            return False

        if section_title == 'Plot':
            current_arc = ''
            for sib in header.find_next_siblings():
                if sib.name == 'h2':
                    break
                if sib.name == 'h3':
                    current_arc = self._clean_text(sib.get_text())
                    into['Plot'][current_arc] = ''
                if sib.name == 'p' and current_arc:
                    chunk = self._clean_text(sib.get_text())
                    if chunk:
                        cur = into['Plot'][current_arc]
                        into['Plot'][current_arc] = (cur + ' ' + chunk).strip()

            return True

        if section_title == 'Equipment':
            for sib in header.find_next_siblings():
                if sib.name == 'h2':
                    break
                if sib.name == 'p':
                    text = sib.get_text(' ', strip=True)
                    name, sep, desc = text.partition(':')
                    if sep:
                        into['Equipment'][name.strip()] = self._clean_text(desc.strip())

            return True

        if section_title == 'Abilities & Powers':
            nen = False
            for sib in header.find_next_siblings():
                if sib.name == 'h2':
                    break
                if sib.name == 'h3':
                    title = sib.get_text(' ', strip=True).lower()
                    nen = ('nen' in title)
                    continue
                if sib.name == 'p':
                    p_text = self._clean_text(sib.get_text())
                    if not p_text:
                        continue
                    if nen:
                        cur = into['Abilities & Powers']['Nen']['Description']
                        into['Abilities & Powers']['Nen']['Description'] = (cur + ' ' + p_text).strip()
                    first_sentence = p_text.split('.')[0]
                    if ':' in first_sentence:
                        key, val = first_sentence.split(':', 1)
                        into['Abilities & Powers'][key.strip()] = self._clean_text(val.strip())
                    else:
                        cur = into['Abilities & Powers']['Description']
                        into['Abilities & Powers']['Description'] = (cur + ' ' + p_text).strip()
                if sib.name == 'table':
                    self._parse_nen_table(sib, into)

            return True

        if section_title == 'Quotes':
            quotes = []
            for sib in header.find_next_siblings():
                if sib.name == 'h2':
                    break
                if sib.name == 'div':
                    ul = sib.find('ul')
                    if not ul:
                        continue
                    quotes.extend(
                        [self._clean_text(' '.join(li.get_text(strip=True).split())) for li in ul.find_all('li')]
                    )
            cleaned = self._clean_quotes(quotes)
            seen, uniq = set(), []
            for q in cleaned:
                if q and len(q) >= 5 and q not in seen:
                    seen.add(q)
                    uniq.append(q)
            into['Quotes'] = uniq

            return True

        for sib in header.find_next_siblings():
            if sib.name == 'h2':
                break
            if sib.name == 'p':
                chunk = self._clean_text(sib.get_text())
                if chunk:
                    cur = into[section_title]
                    into[section_title] = (cur + ' ' + chunk).strip()

        return True

    def _parse_nen_table(self, table, into):
        rows = table.find_all('tr')
        for i, row in enumerate(rows):
            if i == 0:
                continue
            ths = row.find_all('th')
            if len(ths) < 2:
                continue

            raw_type = self._clean_text(ths[0].get_text())
            _, _, nen_type = raw_type.partition(':')
            nen_type = (nen_type.strip() or raw_type).strip()
            nen_name = self._clean_text(ths[1].get_text())
            if not nen_name or not nen_type:
                continue

            description = None
            for sib in row.find_next_siblings('tr'):
                if len(sib.find_all('th')) >= 2:
                    break
                tds = sib.find_all('td')
                if not tds:
                    continue
                best = ''
                for td in tds:
                    if td.find('figure'):
                        continue
                    txt = self._clean_text(td.get_text())
                    if len(txt) > len(best):
                        best = txt
                if best:
                    description = best
                    break

            if description:
                into['Abilities & Powers']['Nen'][nen_name] = {'Description': description, 'Type': nen_type}

    @staticmethod
    def _clean_quotes(quotes):
        out = []
        for q in quotes:
            q = re.sub(r"[♠♥♣]", "", q or "")
            q = q.replace('"', "")
            out.append(" ".join(q.split()).strip())
        return out

    @staticmethod
    def _clean_text(text, strip_parens = True, strip_brackets = True):
        if text is None:
            return None
        text = text.replace('*', '')
        if strip_parens:
            text = re.sub(r'\([^)]*\)', '', text)
        if strip_brackets:
            text = re.sub(r'\[[^\]]*\]', '', text)

        return ' '.join(text.split()).strip()

    @staticmethod
    def _get_age(text):
        if text is None:
            return None
        nums = re.findall(r'\d+', text)

        return nums[-1] if nums else None

    @staticmethod
    def _parse_height(text):
        if not text:
            return {'raw': None}
        
        cm = re.search(r'(\d{2,3})\s*cm', text)
        ftin = re.search(r"(\d)'\s*(\d{1,2})", text)
        out = {'raw': text.strip()}
        if cm:
            out['cm'] = int(cm.group(1))
        if ftin:
            out['ft'] = int(ftin.group(1))
            out['in'] = int(ftin.group(2))

        return out

    @staticmethod
    def _parse_weight(text):
        if not text:
            return {'raw': None}
        kg = re.search(r'(\d{2,3})\s*kg', text)
        lb = re.search(r'(\d{2,3})\s*lb', text)
        out = {'raw': text.strip()}
        if kg:
            out['kg'] = int(kg.group(1))
        if lb:
            out['lb'] = int(lb.group(1))
            
        return out
