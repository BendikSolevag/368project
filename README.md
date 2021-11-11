# Benchmarking NorBert, NbBert and mBert 🤖⚔️

## Requirements

This project requires ```python v3.8``` and ```pip v21.3.1``` <br />
To install the neccesary dependencies, run the ```setup.py```

## Usage

To run the project, run ```python3 main.py``` in the project root.

## Description

This repository aims to benchmark state of the art models for norwegian language modelling in various tasks. These are the benchmarks, their datasets, and the files responsible for performing the testing.


| Benchmark                         | Executable                             | Dataset                                                                                                                                                                                        |
|-----------------------------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sentence-level sentiment polarity | ./sentence_level_sentiment_polarity.py | ./Data/sentence_level_sentiment_polarity/train.json<br>./Data/sentence_level_sentiment_polarity/test.json                                                                                      |
| Dialect classification            | ./DialectClassification.py             | ./Data/dialect_classification/dialect_tweet_train.json<br>./Data/dialect_classification/dialect_tweet_test.json                                                                                |
| Dependency parsing                | ./TokenClassification.py               | ./Data/pos_tagging/no_bokmaal-ud-train.conllu<br>./Data/pos_tagging/no_bokmaal-ud-test.conllu<br>./Data/pos_tagging/no_nynorsk-ud-train.conllu<br>./Data/pos_tagging/no_nynorsk-ud-test.conllu |
| Part-of-speech tagging            | ./TokenClassification.py               | ./Data/pos_tagging/no_bokmaal-ud-train.conllu<br>./Data/pos_tagging/no_bokmaal-ud-test.conllu<br>./Data/pos_tagging/no_nynorsk-ud-train.conllu<br>./Data/pos_tagging/no_nynorsk-ud-test.conllu |
| Named entity recognition          | ./TokenClassification.py               | ./Data/pos_tagging/no_bokmaal-ud-train.conllu<br>./Data/pos_tagging/no_bokmaal-ud-test.conllu<br>./Data/pos_tagging/no_nynorsk-ud-train.conllu<br>./Data/pos_tagging/no_nynorsk-ud-test.conllu |
