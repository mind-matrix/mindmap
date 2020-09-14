import pandas as pd
import re
from bs4 import BeautifulSoup as bs, Tag, NavigableString
from nltk.tokenize import RegexpTokenizer
import json
from bidict import bidict, inverted
from urllib.request import urlopen
from lxml.html import parse

tokenizer = RegexpTokenizer(r'\w+')

def get_title(url):
    return " ".join(url[url.rfind("/")+1 : len(url)].rstrip().split("_"))

def extract(f):
    rows = []
    stats = { 'rels': dict(), 'count': 0, 'mapping': bidict() }
    stats_k = 1
    with open(f, "r", encoding="utf8") as fd:
        for i, line in enumerate(fd):
            if i % 3 == 0:
                try:
                    url = line.split("url=")[1].strip()
                    subj = get_title(url)
                except:
                    print(i)
            elif i % 3 == 1:
                # statement
                soup = bs(line, "html.parser")
                text = []
                tag = []
                stats['count'] += 1
                for segment in soup.contents:
                    if isinstance(segment, Tag):
                        parts = tokenizer.tokenize(segment.text)
                        if segment.has_attr("relation"):
                            if segment["relation"] in stats['rels']:
                                stats['rels'][segment["relation"]] += 1
                            else:
                                stats['rels'][segment["relation"]] = 1
                                stats['mapping'][segment["relation"]] = stats_k
                            tag += [ f'B-{stats_k}' ] + [ f'I-{stats_k}' for i in range(len(parts)-1) ]
                            stats_k += 1
                        else:
                            tag += [ "O" for i in range(len(parts)) ]
                    else:
                        parts = tokenizer.tokenize( str(segment) )
                        tag += [ "O" for i in range(len(parts)) ]
                    text += parts
                rows.append({ 'subject': subj, 'text': text, 'tag': tag })
            else:
                continue
        return pd.DataFrame(rows), stats

def save_stats(stats, figname, mappings):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.autolayout': True})

    fig, axes = plt.subplots(figsize=(7,5), dpi=100)
    plt.bar(stats['rels'].keys(), height=stats['rels'].values())
    plt.xticks(rotation=90)
    fig.savefig(figname)
    with open(mappings+".json", "w") as fd:
        json.dump(dict(stats['mapping']), fd)
    with open(mappings+".inv.json", "w") as fd:
        json.dump(dict(inverted(stats['mapping'])), fd)

def test():
    df_train, stats_train = extract("raw/wikipedia.train")
    df_test, stats_test = extract("raw/wikipedia.test")
    save_stats(stats_train, figname="stats/train.rels.png", mappings="stats/train.map")
    save_stats(stats_test, figname="stats/test.rels.png", mappings="stats/test.map")

if __name__ == "__main__":
    test()