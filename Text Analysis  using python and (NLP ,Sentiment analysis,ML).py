#!/usr/bin/env python
# coding: utf-8

# In[21]:


#pip install textblob


# In[22]:


#pip install spacy


# In[23]:


#!python -m spacy download en_core_web_sm


# In[24]:


#pip install wordcloud


# In[25]:


#pip install scikit-learn


# Text Analysis: Process We Can Follow
# Text Analysis involves various techniques such as text preprocessing, sentiment analysis, named entity recognition, topic modelling, and text classification. Text analysis plays a crucial role in understanding and making sense of large volumes of text data, which is prevalent in various domains, including news articles, social media, customer reviews, and more.
# 
# Below is the process you can follow for the task of Text Analysis as a Data Science professional:
# 
# Gather the text data from various sources.
# Clean and preprocess the text data.
# Convert the text into a numerical format that machine learning algorithms can understand.
# Analyze the text data to gain insights.
# Create relevant features from the text data if necessary.
# Select appropriate NLP models for your task

# In[26]:


import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import spacy
from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
nlp =spacy.load('en_core_web_sm')
data =pd.read_csv(r"C:\Users\User\Downloads\Topic-Modelling\Topic Modelling\articles.csv",encoding='latin-1')
print(data.head())


# The problem we are working on requires us to:
# 
# Create word clouds to visualize the most frequent words in the titles.
# Analyze the sentiment expressed in the articles to understand the overall tone or sentiment of the content.
# Extract named entities such as organizations, locations, and other relevant information from the articles.
# Apply topic modelling techniques to uncover latent topics within the articles.
# Now, let’s move forward by visualizing the word cloud of the titles:

# In[5]:


#comine all titles into single string
titles_text =" ".join(data["Article"])
#print(titles_text)

# creat a Wordcloud abject
wordcloud =WordCloud(width=800,height=400,background_color ="white").generate(titles_text)

# plot the word cloud
fig =px.imshow(wordcloud,title ="word cloud of title")
fig.update_layout(showlegend = False)
fig.show()


# In the above code, we are generating a word cloud based on the titles of the articles. First, we concatenated all the titles into a single continuous string called titles_text using the join method.
# 
# Next, we created a WordCloud object with specific parameters, including the width, height, and background colour, which determine the appearance of the word cloud. Then, we used this WordCloud object to generate the word cloud itself, where the size of each word is proportional to its frequency in the titles

# In[6]:


data['sentiment']=data['Article'].apply(lambda x:TextBlob(x).sentiment.polarity)
#print(data['sentiment'])
fig=px.histogram(data,x='sentiment',title ='Sentiment Distribution')
fig.show()


# In the above code, sentiment analysis is performed on the articles in the dataset to assess the overall sentiment or emotional tone of the articles. The TextBlob library is used here to analyze the sentiment polarity, which quantifies whether the text expresses positive, negative, or neutral sentiment.
# 
# The sentiment.polarity method of TextBlob calculates a sentiment polarity score for each article, where positive values indicate positive sentiment, negative values indicate negative sentiment and values close to zero suggest a more neutral tone. After calculating the sentiment polarities, a histogram is created to visualize the distribution of sentiment scores across the articles.
# 
# Now, let’s perform Named Entity Recognition:

# In[7]:


def extract_named_entities(text):
    doc =nlp(text)
    entities =defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return dict(entities)
data['Named_entities'] =data['Article'].apply(extract_named_entities)
#print(data['Named_entities'])
entity_count =Counter(entity for entities in data['Named_entities'] for entity in entities)
print(entity_count)
entity_df =pd.DataFrame.from_dict(entity_count,orient='index').reset_index()
entity_df.columns =['Entity','Count']
print(entity_df)
fig =px.bar(entity_df.head(10),x="Entity",y="Count",title=" Top 10  names")
fig.show()



# In the above code, we are performing Named Entity Recognition. NER is a natural language processing technique used to identify and extract specific entities such as organizations, locations, names, dates, and more from the text. The extract_named_entities function leverages the spaCy library to analyze each article, identify entities, and categorize them by their respective labels (e.g., “ORG” for organizations, “LOC” for locations). 
# 
# The extracted entities are stored in a new column called Named_Entities in the dataset. Then, a visualization is created to present the top 10 most frequently occurring named entities and their respective counts, allowing for a quick understanding of the prominent entities mentioned in the text data.
# 
# Now, let’s perform Topic Modelling:

# In[9]:


#print(data)


# In[20]:


# Topic modeling
vectorizer =CountVectorizer(max_df =0.95,min_df=2,max_features =1000,stop_words ="english")
tf = vectorizer.fit_transform(data['Article'])
#print(tf)
lda_model =LatentDirichletAllocation(n_components=5,random_state=42)
lda_model_matrix =lda_model.fit_transform(tf)
#print(lda_model_matrix)
topic =['Topic' + str(i) for i in range(lda_model.n_components)]
data['Dominant']=[topic[i]for i in lda_model_matrix.argmax(axis =1)]

fig=px.bar(data['Dominant'].value_counts().reset_index(),x='index',y='Dominant')
fig.show()


# Then we specified parameters such as maximum and minimum document frequency and the maximum number of features (words) to consider, while also removing common English stopwords. 
# 
# Next, we applied LDA using the LatentDirichletAllocation model with five topics as an example. The resulting topic matrix represents each article’s distribution across these five topics. Then, we assign a dominant topic to each article based on the topic with the highest probability, and a bar chart is generated to visualize the distribution of dominant topics across the dataset, providing an overview of the prevalent themes or subjects discussed in the articles.

# In[ ]:




