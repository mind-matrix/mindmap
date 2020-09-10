import pandas as pd
import re
from bs4 import BeautifulSoup as bs, Tag, NavigableString
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

def extract(f):
    rows = []
    with open(f, "r") as fd:
        for i, line in enumerate(fd):
            if (i+1) % 3 == 0:
                continue
            elif (i+1) % 2 == 0:
                # statement
                soup = bs(line, "html.parser")
                text = []
                tag = []
                for segment in soup.contents:
                    if isinstance(segment, Tag):
                        parts = tokenizer.tokenize(segment.text)
                        if segment.has_attr("relation"):
                            tag += [ "B" ] + [ "I" for i in range(len(parts)-2) ] + [ "E" ]
                        else:
                            tag += [ "O" for i in range(len(parts)) ]
                    else:
                        parts = tokenizer.tokenize( str(segment) )
                        tag += [ "O" for i in range(len(parts)) ]
                    text += parts
                rows.append({ 'text': text, 'tag': tag })
        return pd.DataFrame(rows)

def test():
    df = extract("raw/wikipedia.test")
    print(df.head())

if __name__ == "__main__":
    test()