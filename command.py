import requests
import datetime

def send_slack(name, username, msg):
    url = 'https://hooks.slack.com/services/'
    url += 'T0149LRKH9U/B024YSM8UCD/HIwMBKVPGYe5oQ8dO0jUObft'

    payload = {
        'channel': name,
        'username': username,
        'text': msg
    }

    requests.post(url, json=payload)


class BaseRunner(object):
    def __init__(self):
        self.runner = None

    def clear(self):
        self.runner.clear()

    def run(self, args):
        self.runner.start(**args)


class RunTicker(BaseRunner):
    def __init__(self):
        super(RunTicker, self).__init__()
        from ticker import TickerRunner
        self.runner = TickerRunner()


class RunBERTopic(BaseRunner):
    def __init__(self):
        super(RunBERTopic, self).__init__()
        from topic_modeling import BERTopicRunner
        self.runner = BERTopicRunner()


class RunSEC(BaseRunner):
    def __init__(self):
        super(RunSEC, self).__init__()
        from sec import SECRunner
        self.runner = SECRunner()


class RunKeyBERT(BaseRunner):
    def __init__(self):
        super(RunKeyBERT, self).__init__()
        from keyword_extraction import KeyBERTRunner
        self.runner = KeyBERTRunner()


class RunDB(BaseRunner):
    def __init__(self):
        super(RunDB, self).__init__()
        from db import DBRunner
        self.runner = DBRunner()


class CommandManager(object):
    def __init__(self, args):
        self._cmds = []
        getattr(self, args.c)()
        self.args = args

    def runticker(self):
        self.register(RunTicker)

    def runsec(self):
        self.register(RunSEC)

    def runbertopic(self):
        self.register(RunBERTopic)

    def runkeybert(self):
        self.runsec()
        self.register(RunKeyBERT)

    def rundb(self):
        self.register(RunDB)

    def runkermit(self):
        today = datetime.datetime.now().isocalendar()[1] % 2
        if today:
            self.runticker()
            self.runbertopic()
            self.rundb()

    def register(self, cmd_class):
        self._cmds.append(cmd_class())

    def run(self):
        for cmd in self._cmds:
            try:
                cmd.run(vars(self.args))
            except:
                cmd.clear()
                send_slack("#nlp-log", 'News Crawler Bot', 'Something unexpected happened')
