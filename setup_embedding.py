import pandas as pd
import numpy as np

babbage = pd.read_csv('data/text_embeddings/babbage.csv', index_col='id', dtype={'type': str})
curie = pd.read_csv('data/text_embeddings/curie.csv', dtype={'type': str})
train_curie = pd.read_csv('data/text_embeddings/train_curie.csv', index_col='id', dtype={'type': str})
test_curie = pd.read_csv('data/text_embeddings/test_curie.csv', index_col='id', dtype={'type': str})
train_curie['curie_similarity'] = train_curie.curie_similarity.apply(eval).apply(np.array)
test_curie['curie_similarity'] = test_curie.curie_similarity.apply(eval).apply(np.array)

# categorize and resort types
types = [
'sculpture',\
'bag', 'suit', 'doll', 'sheet/linen', 'dish/vessel','jewel', 'tape/ribbon',\
'country',\
'paper','book','magazines', 'album', 'newspaper', 'folder/booklet',\
'invitation',  'calendar',\
'audio recording', 'telegram',\
'packaging', 'crate/box',\
'printed notes', 'small print',\
'seal', 'seal/imprint',\
'letter','letter of honor/honorary address',\
'postcard', 'photo, postcard', 'letter, postcard',\
'manuscript','script, song/vocal music', 'music sheet', 'musical instrument', 'manuscript, musical composition', 'manuscript, sheet music',\
'medal', 'coin', 'label/sign',\
'poster','plan','notes', 'document', 'certificate',\
'graphics', 'drawing', 'design/drawing/sketch','caricature','slide',\
'archaeological find',\
'photo', 'photo negative', 'photographic negative, photographic negative', 'photographic material','digital image'
]