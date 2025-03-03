# from textblob import TextBlob
#
# text = "Thiiis izz a sentnce with erors."
# # df_clean["text"] = df_clean["text"].apply(lambda x: TextBlob(x).correct())
# def sass(s):
#     return str(TextBlob(s).correct())
#
# print(text(lambda x : TextBlob(x).correct()))\

import pandas as pd
from textblob import TextBlob

df = pd.DataFrame({"text": ["Thiiis izz a sentnce.", "Pls fixx thiz messagge."]})

# Apply TextBlob correction
df["corrected_text"] = df["text"].apply(lambda x: str(TextBlob(x).correct()))

print(df)
