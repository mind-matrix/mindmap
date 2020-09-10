import pandas as pd
import re
import spacy
from bs4 import BeautifulSoup as bs
import lxml

def extract_rels(rel):
    rels = []
    with open(rel) as fd:
        for line in fd:
            parts = re.split('\(|,|\)', line)
            rels.append({ 'type': parts[0], 'arg1': parts[1], 'arg2': parts[2], 'order': parts[3] or 'NORMAL' })
    return rels

def extract_sents(xml):
    df = pd.DataFrame(columns=['tokens','h_start','h_end','t_start','t_end','rel','extra_title'])
    with open(xml, "r") as file:
        content = file.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        doc = bs(content, "lxml")
