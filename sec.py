import os
import json

import datetime

import numpy as np

from log import LoggingApp
from edgar_crawler import edgar_crawler, extract_items


class SECRunner(LoggingApp):
    def __init__(self):
        super(SECRunner, self).__init__()

    def change_config(self):
        ticker_list = np.load('./ticker_data/ticker_list.npy')
        with open('./edgar_crawler/config.json') as f:
            data = json.load(f)

        data['edgar_crawler']['cik_tickers'] = list(ticker_list)
        data['edgar_crawler']['end_year'] = datetime.datetime.now().year

        with open('./edgar_crawler/config.json', 'w') as f:
            json.dump(data, f)

    def start(self, **kwargs):
        self.change_config()
        edgar_crawler.main(self.logger)
        extract_items.main(self.logger)
