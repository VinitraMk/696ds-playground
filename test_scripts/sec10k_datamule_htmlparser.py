from datamule import Document
from datamule import Portfolio
import json
from time import time
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import re
import os

class SECFileParser:

    def __init__(self, filename, filetype = "10K"):
        self.filename = filename
        self.filetype = filetype
        with open(f'data/SEC-10K/{self.filename}.htm', 'r') as f:
            self.full_html_content = f.read()
        self.page_soup = BeautifulSoup(self.full_html_content, 'html.parser')
        self.page_data = {}
        self.sec10k_item_names = ['item1', 'item1a', 'item1b', 'item1c', 'item2', 'item3', 'item4','item5', 'item6', 'item7',
            'item7a', 'item8', 'item9', 'item9a', 'item9b', 'item9c', 'item10', 'item11', 'item12', 'item13', 'item14',
            'item15', 'item16', 'signatures']
        self.sec10k_page_keys = {
            'item1': 'parti',
            'item1a': 'parti',
            'item1b': 'parti',
            'item1c': 'parti',
            'item2': 'parti',
            'item3': 'parti',
            'item4': 'parti',
            'item5': 'partii',
            'item6': 'partii',
            'item7': 'partii',
            'item7a': 'partii',
            'item8': 'partii',
            'item9': 'partii',
            'item9a': 'partii',
            'item9b': 'partii',
            'item9c': 'partii',
            'item10': 'partiii',
            'item11': 'partiii',
            'item12': 'partiii',
            'item13': 'partiii',
            'item14': 'partiii',
            'item15': 'partiv',
            'item16': 'partiv',
            'signatures': 'partiv'
        }

    def save2json(self):
        '''
        document = Document(filename=Path(f'data/SEC-10K/{self.filename}.htm'), type=self.filetype)
        document.parse()
        with open(f'data/SEC-10K/parsed_data/datamule/parsed-datamule-{self.filename}.json', 'w', encoding='utf-8') as f:
            json.dump(document.data, f, indent = 4)
        '''

        with open(f'data/SEC-10K/parsed_data/custom/parsed-{self.filename}.json', 'w', encoding = 'utf-8') as f:
            json.dump(self.page_data, f, indent = 4)

    def build_page_json(self):
        '''
        a_names = [re.sub(r"\s+", " ", alink.string).replace(" ", "").replace(".", "").lower() for alink in a_links]
        a_names = ["item1c" if s == "i" else s for s in a_names]
        #print(a_names)
        self.page_keys = {}
        curr_part = a_names[0]
        for an in a_names:
            if an.startswith('part'):
                self.page_data[an] = {}
                curr_part = an
            else:
                self.page_data[curr_part][an] = {}
                self.page_keys[an] = curr_part
        #print(self.page_data)
        '''
        for it in self.sec10k_item_names:
            part_key = self.sec10k_page_keys[it]
            if part_key in self.page_data:
                self.page_data[part_key][it] = {}
            else:
                self.page_data[part_key] = {}
                self.page_data[part_key][it] = {}

    def parse_toc(self):
        
        #soup = BeautifulSoup(self.full_html_content, 'html.parser')
        toc_atag = self.page_soup.find('a', string = 'Table of Contents')
        #print('a tag', toc_atag)
        toc_id = toc_atag['href'].replace('#', '')
        print(toc_id)
        toc_div = self.page_soup.find('div', { 'id': toc_id })
        nel = toc_div
        #print(toc_div)
        while True:
            nel = nel.next_sibling
            #print('next el', nel)
            toc_table_el = nel.find('table')
            if toc_table_el != None:
                #print('table el:', toc_table_el)
                break
        trs = toc_table_el.find_all('tr')
        first_tds = [row.find('td') for row in trs]
        a_links = [cell.find('a') for cell in first_tds]
        a_links = [alink for alink in a_links if alink != None]
        self.a_items = [alink for alink in a_links if (alink.string.startswith('I') or alink.string.startswith('S'))]
        self.a_item_names = [re.sub(r"\s+", " ", aitem.string.replace(' ', '')).strip().replace(" ", "").replace(".", "").lower() for aitem in self.a_items]
        #print(sec_items)
        self.item_ids = [aitem['href'].replace('#', '') for aitem in self.a_items]
        print('Item ids', self.item_ids)
        self.a_item_names = ["item1c" if s == "i" else s for s in self.a_item_names]
        print('item names', self.a_item_names)
        self.build_page_json() 


    def get_tables(self, el):
        #print('el type', type(el), el)
        tbls = pd.read_html(str(el))
        return tbls
        
    def extract_data(self, section_els, section_key):
        str_data = ''
        tables = []
        for el in section_els:
            #print(list(el.children))
            span_els = [child for child in el.children if child.name == "span"]
            #span_els = el.find_all('span', recursive = True)
            if len(span_els) == 1:
                #is_table_present, tbls = self.is_table_present(span_els[0])
                if span_els[0].string and span_els[0].string != 'Table of Contents' and not(span_els[0].string.isnumeric()):
                    #print(span_els[0].string)
                    str_data += span_els[0].string + "\n"
            elif len(span_els) > 1:
                for el in span_els:
                    #print(el.string)
                    #is_table_present, tbls = self.is_table_present(el)
                    if el.string and el.string != 'Table of Contents' and not(el.string.isnumeric()):
                        str_data += el.string + " "
                        
            #tbl_els = el.find_all('table', recursive = True)
            tbl_els = [child for child in el.children if child.name == 'table']
            if len(tbl_els) > 0:
                #print('table el: ---->', el)
                tbls = self.get_tables(el)
                for i, tbl in enumerate(tbls):
                    str_data += "\n#TABLE_BEGIN#\n"
                    str_data += f"TABLE_ID: {i}\n"
                    str_data += "\n#TABLE_END#\n"
                    mask = (tbl.iloc[1:, :].isna()).all(axis=0)
                    tbl_df = tbl.drop(tbl.columns[mask], axis=1).fillna('')
                    #print('table el ------>', tbl_df)
                    #str_data += tbl_df.to_string()
                    json_str = tbl_df.to_json(orient = 'records')
                    json_obj = json.loads(json_str)
                    tables.append(json_obj)
            #print(str_data)
        #print(str_data)
        self.page_data[self.sec10k_page_keys[section_key]][section_key] = {
            'text': str_data,
            'tables': tables
        }
        #print(self.page_data[self.page_keys[section_key]])

    def parse_section(self, section_idx):
        #print(len(self.item_ids), section_idx)
        section_id = self.item_ids[section_idx]
        next_section_id = self.item_ids[section_idx+1] if section_idx+1 < len(self.item_ids) else -1
        #print('sids', section_id, next_section_id)
        if next_section_id != -1:
            next_el = self.page_soup.find('div', {'id': section_id})
            end_el = self.page_soup.find('div', {'id': next_section_id})
            #print('next el', next_el)
            section_els = []
            #section_els.append(next_el)
            while next_el and next_el != end_el:
                #print(next_el)
                section_els.append(next_el)
                next_el = next_el.find_next_sibling()

            #print(section_els)
            self.extract_data(section_els, self.a_item_names[section_idx])


if __name__ == "__main__":

    #filename = '000000248824000012-amd-20231230'
    sec_filing_list = [fn.replace(".htm", "") for fn in os.listdir('data/SEC-10K')]

    for filename in sec_filing_list:
        #filename = '000000248824000012-amd-20231230'
        print(f'\nParsing file: {filename}')
        sec_parser = SECFileParser(filename)
        #sec_parser.save2json()
        sec_parser.parse_toc()
        for si in range(len(sec_parser.item_ids)-1):
            sec_parser.parse_section(si)
        sec_parser.save2json()
        break


    '''
    portfolio = Portfolio("data/SEC-10K-datamule-test")

    for ten_k in portfolio.document_type("10-K"):
        ten_k.parse()
        with open(f'data/SEC-10K-datamule-test/parsed-datamule-{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(ten_k.data, f, indent=4)  
    '''