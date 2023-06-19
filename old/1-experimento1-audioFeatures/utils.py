import pandas as pd

import pandas as pd


def list_duplicates(seq):
  seen = set()
  seen_add = seen.add
  # adds all elements it doesn't know yet to seen and all other to seen_twice
  seen_twice = set( x for x in seq if x in seen or seen_add(x) )
  # turn the set into a list (as requested)
  return list( seen_twice )


meta = pd.read_csv('processed_results.csv')
ids = meta['voiceID'].tolist()

print(ids[1])
a = ['a','a']
print(list_duplicates(a))

df = pd.read_csv('processed_results.csv')
df2 = df.dropna(axis=0 , how='any')
df2.to_csv("sex_dataset_2.csv", index=False)