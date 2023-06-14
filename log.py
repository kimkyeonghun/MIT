import os

import datetime
import logging
import requests


class LoggingApp(object):
    logger_name = None

    def __init__(self):
        super(LoggingApp, self).__init__()
        self._logger = None
        self.log_path = None

    @property
    def logger(self):
        if self._logger is None:
            self._logger = self.get_logger(self.logger_name)
            # self.logger_name = self.
        return self._logger

    def get_logger(self, logger_name, log_path='./logs'):
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        logger = logging.getLogger(logger_name)

        if len(logger.handlers) > 0:
            return logger

        date_format = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(
            '[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s %(message)s', date_format)
        i = 0
        today = datetime.datetime.now()
        name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d' % i+'.log'
        while os.path.exists(os.path.join(log_path, name)):
            i += 1
            name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d' % i+'.log'

        fileHandler = logging.FileHandler(os.path.join(log_path, name))
        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)

        logger.setLevel(logging.INFO)
        logger.info('Writing logs at {}'.format(os.path.join(log_path, name)))
        return logger

    def send_slack(self, name, username, msg):
        url = 'https://hooks.slack.com/services/'
        

        payload = {
            'channel': name,
            'username': username,
            'text': msg
        }

        requests.post(url, json=payload)

    def clear(self):
        self.logger.exception("ERROR")
        self.logger.handlers.clear()
        logging.shutdown()
