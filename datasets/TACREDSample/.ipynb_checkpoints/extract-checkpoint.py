import pandas as pd
import json

mappings = {
    'token': 'tokens',
    'subj_start': 'h_start',
    'subj_end': 'h_end',
    'obj_start': 't_start',
    'obj_end': 't_end',
    'relation': 'rel',
    'stanford_pos': 'extra_pos',
    'stanford_ner': 'extra_ner',
    'stanford_head': 'extra_head',
    'stanford_deprel': 'extra_deprel'
}

def extract(file):
    rows = []
    with open(file, "r") as fd:
        records = json.load(fd)
        for i, record in enumerate(records):
            rows.append({ mappings[k]: v for k, v in record.items() if k in mappings.keys() })
        return pd.DataFrame.from_dict(rows, orient='columns')

def test():
    df = extract("raw/dev.json")
    print(df.head())

if __name__ == "__main__":
    test()