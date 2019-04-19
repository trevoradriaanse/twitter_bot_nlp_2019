import argparse
import sys
import random

import newspaper

def get_article_keywords(news_source='http://cnn.com/'):
    keywords = []
    news_source = newspaper.build(news_source, memoize_articles=False)
    paper = random.choice(news_source.articles)
    paper.download()
    paper.parse()
    paper.nlp()
    keywords.extend(paper.summary.split())
    return keywords