## CTM(Concept-Ticker Mathcing Algorithm)

Concept들에 가장 연관 있는 Ticker를 매칭하는 알고리즘을 개발함. 기존 Kensho에서 제공했던 것을 Fint 기술로 내재화하는 것을 목표로 하고 있음.
CTM은 BERTopic과 Zero-shot Classification을 사용하여 각 Ticker에 가장 연관 있는 Concept을 매칭 시킬 수 있다. News, SEC, ETF로 각 데이터에 CTM을 적용하도록 한다.

### Environment

- python 3.8
- bertopic 0.11.0
- keybert 0.6.0
- sec-cik-mapper 2.1.0
- sentence-transformers 2.2.0
- pytorch 1.7.0
- transformers 4.19.2
- edgar-crawler

## Install

### News
News 데이터에 대해 BERTopic과 Zero-shot Classification을 사용한다.
따라서 bertopic, sentence-trasnformers, transformers, pytorch가 필수적으로 설치 되어야 한다.

### SEC
SEC 데이터를 수집하기 위해 edgar-crawler를 사용해야 한다. 설치 및 사용 방법은 아래 링크를 참조하면 된다.

Install and Usage: [edgar-crawler](https://github.com/nlpaueb/edgar-crawler)
logging 기능을 사용하기 위해서 코드를 일부 수정해야 한다.

### ETF?




