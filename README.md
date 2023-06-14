## MIT(Matching Industry to Theme)

테마(Theme)들에 가장 연관 있는 Industry를 매칭하는 알고리즘을 개발함.
MIT는 BERTopic과 Zero-shot Classification을 사용하여 각 Industry에 가장 연관 있는 테마(Theme)을 매칭 시킬 수 있다. News, SEC, ETF로 각 데이터에 MIT을 적용하도록 한다.

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




