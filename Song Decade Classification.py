################################################################################
# Matthew Spitulnik ############################################################
# Natural Language Processing ##################################################
# Song Decade Classification ###################################################
# Project Summary: For this project, the goal was to perform some kind of classification 
# task using different variations of feature sets to compare the results. I continued 
# working with song lyric data that I collected in previous projects; Comparing Corpora 
# with Corpus Statistics (CCCS) and Sentiment and Exploratory Analysis (SEA), both 
# of which can also be found within this project respository. I adjusted the song 
# collection and lyric collection Python scripts to collect the song lyrics for 
# Billboard top 100 songs, each year, going back to the 1950’s. Using different 
# modeling techniques and feature-set-based variations, I wanted to try to build 
# a model that could correctly identify the decade a song was written in based on 
# its lyrics. The following NLP techniques were applied to create different feature 
# sets that would be used to build the models: frequency distributions/document 
# term matrices, sentiment/lexicon categorical analysis, and POS tagging/stop word 
# differentiation. I created my prediction algorithms using cross validation with 
# Naïve Bayes, Multinomial Naïve Bayes, Precision/Recall/F1 scores, and Support 
# Vector Machines (SVM). I also used the “TopK” variation option in the scikit-learn 
# Python package, which looks at the top “K” number of predictions, and if the 
# prediction by the model is in the top K, it considers the prediction to be correct.
################################################################################

################################################################################
### Install and load required packages #########################################
################################################################################

###Install the required libraries.
#%pip install spacy
#%pip install liwc
#%pip install liwc-text-analysis
#%pip install empath
#%pip install html5lib
#%pip install inflect
#%pip install contractions
#%pip install spacy_cleaner
#%pip install pandas
#%pip install re
#%pip install requests
#%pip install nltk
#%pip install lxml
#%pip install bs4
#%pip install sys
#%pip install gensim
#%pip install random
#%pip install json
#%pip install scikit-learn
#%pip install matplotlib
#%pip install numpy

###Import the required libraries
import spacy
#Make sure to run this below command in an outside command prompt:
#python -m spacy download en_core_web_sm
sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words
import spacy_cleaner
from spacy_cleaner.processing import removers, replacers, mutators
import pandas as pd
import re
import requests
import html5lib
import nltk
import sys
#!{sys.executable} -m pip install contractions
import contractions
import inflect
wordPlur=inflect.engine()
from empath import Empath
lexicon = Empath()
import liwc
import gensim
import random
import json
import sklearn
import matplotlib.pyplot as plt
import numpy

from nltk import FreqDist
from lxml import html
from urllib.request import urlopen
from urllib import request
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
from nltk.collocations import *
from nltk import sent_tokenize
nltk.download('vader_lexicon',quiet=True)
nltk.download('stopwords',quiet=True)
stopwords=nltk.corpus.stopwords.words('english')
nltk.download('wordnet',quiet=True)
nltk.download('omw-1.4',quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
from liwc import liwc
from collections import Counter
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import top_k_accuracy_score
from sklearn import covariance
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

################################################################################
### Import the data sets and other files that will be used throughout the code #
################################################################################

###This creates a "default_directory" variable, where the directory path to the
# data files folder containing all of the required data sets is saved so that it
# does not need to be constantly re-entered. Remember to use forward slashes
# instead of back slashes in the directory path. For example, if the datafiles
# folder is saved in "C:\home\project\datafiles", then "C:/home/project"
# would be inserted between the quotes below.
default_directory = "<DEFAULT DIRECTORY PATH HERE>"

#load the data frames that were either created in CCCS, SEA, or that will be created later so that the web scraping or freq dist scripts don't need to be run and waited for
songArtDF = pd.read_csv(f'{default_directory}/datafiles/songArtDF.csv')
songArtDFOrg = pd.read_csv(f'{default_directory}/datafiles/songArtDFOrg.csv')
songArtLyrics =pd.read_csv(f'{default_directory}/datafiles/songArtLyrics.csv')
pre80_lyrics =pd.read_csv(f'{default_directory}/datafiles/pre80_lyrics.csv')
songArtDF_pre80 =pd.read_csv(f'{default_directory}/datafiles/songArtDF_pre80.csv')
FullSongDF_pre80 =pd.read_csv(f'{default_directory}/datafiles/FullSongDF_pre80.csv')
pre80ORG =pd.read_csv(f'{default_directory}/datafiles/pre80ORG.csv')
pre80Lyrics =pd.read_csv(f'{default_directory}/datafiles/pre80Lyrics.csv')
fullLyrList =pd.read_csv(f'{default_directory}/datafiles/fullLyrList.csv')
fullLyrBCU =pd.read_csv(f'{default_directory}/datafiles/fullLyrBCU.csv')
fullLyrFreqCU =pd.read_csv(f'{default_directory}/datafiles/fullLyrFreqCU.csv')
LyrFreqDF =pd.read_csv(f'{default_directory}/datafiles/LyrFreqDF.csv')
LyrFreqDF_2=pd.read_csv(f'{default_directory}/datafiles/LyrFreqDF_2.csv')
LyrSentDF=pd.read_csv(f'{default_directory}/datafiles/LyrSentDF.csv')
LyrEmpDF =pd.read_csv(f'{default_directory}/datafiles/LyrEmpDF.csv')
LyrLiwcDF=pd.read_csv(f'{default_directory}/datafiles/LyrLiwcDF.csv')
SentEmpDFLiwc=pd.read_csv(f'{default_directory}/datafiles/SentEmpDFLiwc.csv')
POScleanUp=pd.read_csv(f'{default_directory}/datafiles/POScleanUp.csv')
LyrPOSDF=pd.read_csv(f'{default_directory}/datafiles/LyrPOSDF.csv')
LyrPOSDF_NLTK=pd.read_csv(f'{default_directory}/datafiles/LyrPOSDF_NLTK.csv')
LyrPOSDF_SPC =pd.read_csv(f'{default_directory}/datafiles/LyrPOSDF_SPC.csv')
LyrPOSDF_GM =pd.read_csv(f'{default_directory}/datafiles/LyrPOSDF_GM.csv')

#load the liwc file that will be required for the liwc section of the analysis
liwc = liwc.Liwc(f'{default_directory}/datafiles/liwccombo2007.txt')

#load the first json file that will be used in the project
with open(f'{default_directory}/json_files/feature_set_freqDist.json', 'r') as file:
    feature_set_freqDist = json.load(file)

for i in range(0,len(feature_set_freqDist)):
    feature_set_freqDist[i]=tuple(feature_set_freqDist[i])

file.close()

#load the next json file that will be used in the project
with open(f'{default_directory}/json_files/feature_set_freqDist_2.json', 'r') as file:
    feature_set_freqDist_2 = json.load(file)

for i in range(0,len(feature_set_freqDist_2)):
    feature_set_freqDist_2[i]=tuple(feature_set_freqDist_2[i])

file.close()

#load the final json file that will be used in the project
with open(f'{default_directory}/json_files/feature_set_ALL.json', 'r') as file:
    feature_set_ALL = json.load(file)

for i in range(0,len(feature_set_ALL)):
    feature_set_ALL[i]=tuple(feature_set_ALL[i])

file.close()

################################################################################
### Collect Additional Required Data ###########################################
################################################################################

#original data set I worked with was from 1980 to 2019. Adding songs from the 1960's through 1979.
# I technically could also get songs from 46-49, 20-22, but I want a full decades worth of data where possible.
# In the 50's, top 100 songs weren't used until 1959. An additional song collection algorithm for that will be used in a second.
songArtDF_pre80=pd.DataFrame()
for i in range(1960,1980):
    songArtyear = "https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_" + str(i)
    temptables=pd.read_html(songArtyear)
    for h in range(0,len(temptables)):
        if 'Title' and 'Artist(s)' in temptables[h]:
            songArttable=temptables[h][['Title','Artist(s)']]
    songArttable['Year']=i

    CombDF=[songArtDF_pre80,songArttable]
    songArtDF_pre80=pd.concat(CombDF)

songArtDF_pre80=songArtDF_pre80.reset_index(drop=True)

#now create song collector for songs in the 50's, which requires some additional qualifiers.
songArtDF_50=pd.DataFrame()
for i in range(1950,1960):
    if i <= 1955:
        songArtyear = "https://en.wikipedia.org/wiki/Billboard_year-end_top_30_singles_of_" + str(i)
    elif i > 1955 and i < 1959:
        songArtyear = "https://en.wikipedia.org/wiki/Billboard_year-end_top_50_singles_of_" + str(i)
    elif i == 1959:
        songArtyear = "https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_1959"
    temptables=pd.read_html(songArtyear)
    songArttable=temptables[0][['Title','Artist(s)']]
    songArttable['Year']=i

    CombDF=[songArtDF_50,songArttable]
    songArtDF_50=pd.concat(CombDF)

songArtDF_50=songArtDF_50.reset_index(drop=True)

songArtDF_50

songArtDF_pre80

songArtDF_pre80.to_csv(f'{default_directory}/datafiles/songArtDF_pre80.csv',index=False)

#1960-1979 should have 2000 songs, but has 2001, figuring out where the issue is
pre80_pivot=songArtDF_pre80.pivot_table(columns=['Year'], aggfunc ='size')
pre80_pivot #output shows that the year 1969 has 101 songs instead of 100
#####It turns out two songs tied for 100th place in 1969, so both songs will be used.

#the 50's decade should have 430 songs: 6 years with 30 songs, 3 years with 50 songs, 1 year with 100 songs, for a total of 430. Seeing why there is 431 instead.
dec50_pivot=songArtDF_50.pivot_table(columns=['Year'], aggfunc ='size')
dec50_pivot #output shows that the year 1958 has 51 songs instead of 50
#####It turns out two songs tied for 50th place in 1958, so both songs will be used.

#combine all of the data into one dataframe for list of songs before 1980
CombDF=[songArtDF_50,songArtDF_pre80]
FullSongDF_pre80=pd.concat(CombDF)

FullSongDF_pre80=FullSongDF_pre80.reset_index(drop=True)
FullSongDF_pre80

FullSongDF_pre80.to_csv(f'{default_directory}/datafiles/FullSongDF_pre80.csv',index=False)

#create copy of the DF with artists before adding to it
pre80_lyrics=FullSongDF_pre80.copy(deep=True)

#now add the lyrics for the songs before the 1980s
for i in pre80_lyrics.index:
    theremoval=re.sub('^The ','',pre80_lyrics.loc[i,'Artist(s)'])
    firstletter=theremoval[0].lower()
    artnospace= re.sub('[\W]','',theremoval)
    titlenospace= re.sub('[\W]','',pre80_lyrics.loc[i,'Title'])
    templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
    templyricURL=templyricURL.lower()

    linktest = requests.get(templyricURL)
    if linktest.status_code == 200:
        templyrichtml=requests.get(templyricURL)
        templyriccontent=html.fromstring(templyrichtml.content)
        templyric=str(templyriccontent.xpath('//*[@id="ldata"]/div[3]/text()'))
        pre80_lyrics.loc[i,'Lyrics']=templyric
    else:
        thetest = re.match('^The ', pre80_lyrics.loc[i,'Artist(s)'])
        if thetest:
            artnospace= re.sub('[\W]','',pre80_lyrics.loc[i,'Artist(s)'])
            templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
            templyricURL=templyricURL.lower()

            linktest = requests.get(templyricURL)
            if linktest.status_code == 200:
                templyrichtml=requests.get(templyricURL)
                templyriccontent=html.fromstring(templyrichtml.content)
                templyric=str(templyriccontent.xpath('//*[@id="ldata"]/div[3]/text()'))
                pre80_lyrics.loc[i,'Lyrics']=templyric
            else:
                if ' and ' in pre80_lyrics.loc[i,'Artist(s)']:
                    andsubremoval=re.sub(' and (?<= and ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
                else:
                    andsubremoval=pre80_lyrics.loc[i,'Artist(s)']
                if ' & ' in andsubremoval:
                    ampsubremoval=re.sub(' & (?<= & ).*$','',andsubremoval)
                else:
                    ampsubremoval=andsubremoval
                if ' featuring ' in ampsubremoval:
                    featsubremoval=re.sub(' featuring (?<= featuring ).*$','',ampsubremoval)
                else:
                    featsubremoval=ampsubremoval
                    artnospace= re.sub('[\W]','',featsubremoval)
        else:
            if ' and ' in pre80_lyrics.loc[i,'Artist(s)']:
                andsubremoval=re.sub(' and (?<= and ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
            else:
                andsubremoval=pre80_lyrics.loc[i,'Artist(s)']
            if ' & ' in andsubremoval:
                ampsubremoval=re.sub(' & (?<= & ).*$','',andsubremoval)
            else:
                ampsubremoval=andsubremoval
            if ' featuring ' in ampsubremoval:
                featsubremoval=re.sub(' featuring (?<= featuring ).*$','',ampsubremoval)
            else:
                featsubremoval=ampsubremoval
                artnospace= re.sub('[\W]','',featsubremoval)
        templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
        templyricURL=templyricURL.lower()
        templyrichtml=requests.get(templyricURL)
        templyriccontent=html.fromstring(templyrichtml.content)
        templyric=str(templyriccontent.xpath('//*[@id="ldata"]/div[3]/text()'))
        pre80_lyrics.loc[i,'Lyrics']=templyric

#a large number of songs didn't work, the xpath to the song lyric text appears to be different for many of those songs, adjusting script to fill in the lyrics that didn't work
for i in pre80_lyrics.index:
    if pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\', \'\\n\\n\']' or pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\', \'\\n\']' or pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\', \'\\n\']':
        theremoval=re.sub('^The ','',pre80_lyrics.loc[i,'Artist(s)'])
        firstletter=theremoval[0].lower()
        artnospace= re.sub('[\W]','',theremoval)
        titlenospace= re.sub('[\W]','',pre80_lyrics.loc[i,'Title'])
        templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
        templyricURL=templyricURL.lower()

        linktest = requests.get(templyricURL)
        if linktest.status_code == 200:
            templyrichtml=requests.get(templyricURL)
            templyriccontent=html.fromstring(templyrichtml.content)
            templyric=str(templyriccontent.xpath('//*[@id="sbmtlyr"]/text()'))
            pre80_lyrics.loc[i,'Lyrics']=templyric
        else:
            thetest = re.match('^The ', pre80_lyrics.loc[i,'Artist(s)'])
            if thetest:
                artnospace= re.sub('[\W]','',pre80_lyrics.loc[i,'Artist(s)'])
                templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
                templyricURL=templyricURL.lower()

                linktest = requests.get(templyricURL)
                if linktest.status_code == 200:
                    templyrichtml=requests.get(templyricURL)
                    templyriccontent=html.fromstring(templyrichtml.content)
                    templyric=str(templyriccontent.xpath('//*[@id="sbmtlyr"]/text()'))
                    pre80_lyrics.loc[i,'Lyrics']=templyric
                else:
                    if ' and ' in pre80_lyrics.loc[i,'Artist(s)']:
                        andsubremoval=re.sub(' and (?<= and ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
                    else:
                        andsubremoval=pre80_lyrics.loc[i,'Artist(s)']
                    if ' & ' in andsubremoval:
                        ampsubremoval=re.sub(' & (?<= & ).*$','',andsubremoval)
                    else:
                        ampsubremoval=andsubremoval
                    if ' featuring ' in ampsubremoval:
                        featsubremoval=re.sub(' featuring (?<= featuring ).*$','',ampsubremoval)
                    else:
                        featsubremoval=ampsubremoval
                    artnospace= re.sub('[\W]','',featsubremoval)
            else:
                if ' and ' in pre80_lyrics.loc[i,'Artist(s)']:
                    andsubremoval=re.sub(' and (?<= and ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
                else:
                    andsubremoval=pre80_lyrics.loc[i,'Artist(s)']
                if ' & ' in andsubremoval:
                    ampsubremoval=re.sub(' & (?<= & ).*$','',andsubremoval)
                else:
                    ampsubremoval=andsubremoval
                if ' featuring ' in ampsubremoval:
                    featsubremoval=re.sub(' featuring (?<= featuring ).*$','',ampsubremoval)
                else:
                    featsubremoval=ampsubremoval
                artnospace= re.sub('[\W]','',featsubremoval)
            templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
            templyricURL=templyricURL.lower()
            templyrichtml=requests.get(templyricURL)
            templyriccontent=html.fromstring(templyrichtml.content)
            templyric=str(templyriccontent.xpath('//*[@id="sbmtlyr"]/text()'))
            pre80_lyrics.loc[i,'Lyrics']=templyric
    else:
        if pre80_lyrics.loc[i,'Lyrics']=='[]':
            theremoval=re.sub('^The ','',pre80_lyrics.loc[i,'Artist(s)'])
            firstletter=theremoval[0].lower()
            artnospace= re.sub('[\W]','',theremoval)
            titlenospace= re.sub('[\W]','',pre80_lyrics.loc[i,'Title'])
            templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
            templyricURL=templyricURL.lower()

            linktest = requests.get(templyricURL)
            if linktest.status_code == 200:
                templyrichtml=requests.get(templyricURL)
                templyriccontent=html.fromstring(templyrichtml.content)
                templyric=str(templyriccontent.xpath('//*[@id="ldata"]/div[4]/text()'))
                pre80_lyrics.loc[i,'Lyrics']=templyric
            else:
                thetest = re.match('^The ', pre80_lyrics.loc[i,'Artist(s)'])
                if thetest:
                    artnospace= re.sub('[\W]','',pre80_lyrics.loc[i,'Artist(s)'])
                    templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
                    templyricURL=templyricURL.lower()

                    linktest = requests.get(templyricURL)
                    if linktest.status_code == 200:
                        templyrichtml=requests.get(templyricURL)
                        templyriccontent=html.fromstring(templyrichtml.content)
                        templyric=str(templyriccontent.xpath('//*[@id="ldata"]/div[4]/text()'))
                        pre80_lyrics.loc[i,'Lyrics']=templyric
                    else:
                        if ' and ' in pre80_lyrics.loc[i,'Artist(s)']:
                            andsubremoval=re.sub(' and (?<= and ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
                        else:
                            andsubremoval=pre80_lyrics.loc[i,'Artist(s)']
                        if ' & ' in andsubremoval:
                            ampsubremoval=re.sub(' & (?<= & ).*$','',andsubremoval)
                        else:
                            ampsubremoval=andsubremoval
                        if ' featuring ' in ampsubremoval:
                            featsubremoval=re.sub(' featuring (?<= featuring ).*$','',ampsubremoval)
                        else:
                            featsubremoval=ampsubremoval
                        artnospace= re.sub('[\W]','',featsubremoval)
                else:
                    if ' and ' in pre80_lyrics.loc[i,'Artist(s)']:
                        andsubremoval=re.sub(' and (?<= and ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
                    else:
                        andsubremoval=pre80_lyrics.loc[i,'Artist(s)']
                    if ' & ' in andsubremoval:
                        ampsubremoval=re.sub(' & (?<= & ).*$','',andsubremoval)
                    else:
                        ampsubremoval=andsubremoval
                    if ' featuring ' in ampsubremoval:
                        featsubremoval=re.sub(' featuring (?<= featuring ).*$','',ampsubremoval)
                    else:
                        featsubremoval=ampsubremoval
                    artnospace= re.sub('[\W]','',featsubremoval)
                templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
                templyricURL=templyricURL.lower()
                templyrichtml=requests.get(templyricURL)
                templyriccontent=html.fromstring(templyrichtml.content)
                templyric=str(templyriccontent.xpath('//*[@id="ldata"]/div[4]/text()'))
                pre80_lyrics.loc[i,'Lyrics']=templyric

#still didn't get a bunch of songs, realized the syntax for some of the if statements were incorrect, corrected those
for i in range(0,len(pre80_lyrics)):
    if pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\', \'\\n\\n\']' or pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\', \'\\n\']' or pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\', \'\\n\']':
        theremoval=re.sub('^The ','',pre80_lyrics.loc[i,'Artist(s)'])
        firstletter=theremoval[0].lower()
        artnospace= re.sub('[\W]','',theremoval)
        titlenospace= re.sub('[\W]','',pre80_lyrics.loc[i,'Title'])
        templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
        templyricURL=templyricURL.lower()

        linktest = requests.get(templyricURL)
        if linktest.status_code == 200:
            templyrichtml=requests.get(templyricURL)
            templyriccontent=html.fromstring(templyrichtml.content)
            templyric=str(templyriccontent.xpath('//*[@id="sbmtlyr"]/text()'))
            pre80_lyrics.loc[i,'Lyrics']=templyric
            print(1)
        else:
            thetest = re.match('^The ', pre80_lyrics.loc[i,'Artist(s)'])
            if thetest:
                artnospace= re.sub('[\W]','',pre80_lyrics.loc[i,'Artist(s)'])
                templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
                templyricURL=templyricURL.lower()

                linktest = requests.get(templyricURL)
                if linktest.status_code == 200:
                    templyrichtml=requests.get(templyricURL)
                    templyriccontent=html.fromstring(templyrichtml.content)
                    templyric=str(templyriccontent.xpath('//*[@id="sbmtlyr"]/text()'))
                    pre80_lyrics.loc[i,'Lyrics']=templyric
                    print(2)
                else:
                    if ' and ' in pre80_lyrics.loc[i,'Artist(s)']:
                        andsubremoval=re.sub(' and (?<= and ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
                    else:
                        andsubremoval=pre80_lyrics.loc[i,'Artist(s)']
                    if ' & ' in andsubremoval:
                        ampsubremoval=re.sub(' & (?<= & ).*$','',andsubremoval)
                    else:
                        ampsubremoval=andsubremoval
                    if ' featuring ' in ampsubremoval:
                        featsubremoval=re.sub(' featuring (?<= featuring ).*$','',ampsubremoval)
                    else:
                        featsubremoval=ampsubremoval
                    artnospace= re.sub('[\W]','',featsubremoval)
            else:
                if ' and ' in pre80_lyrics.loc[i,'Artist(s)']:
                    andsubremoval=re.sub(' and (?<= and ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
                else:
                    andsubremoval=pre80_lyrics.loc[i,'Artist(s)']
                if ' & ' in andsubremoval:
                    ampsubremoval=re.sub(' & (?<= & ).*$','',andsubremoval)
                else:
                    ampsubremoval=andsubremoval
                if ' featuring ' in ampsubremoval:
                    featsubremoval=re.sub(' featuring (?<= featuring ).*$','',ampsubremoval)
                else:
                    featsubremoval=ampsubremoval
                artnospace= re.sub('[\W]','',featsubremoval)
            templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
            templyricURL=templyricURL.lower()
            templyrichtml=requests.get(templyricURL)
            templyriccontent=html.fromstring(templyrichtml.content)
            #templyric=str(templyriccontent.xpath('//*[@id="sbmtlyr"]/text()'))
            templyric=str(templyriccontent.xpath('//*[@id="ldata"]/div[4]/text()'))
            pre80_lyrics.loc[i,'Lyrics']=templyric
            print(3)
    else:
        if pre80_lyrics.loc[i,'Lyrics']=='[]':
            theremoval=re.sub('^The ','',pre80_lyrics.loc[i,'Artist(s)'])
            firstletter=theremoval[0].lower()
            artnospace= re.sub('[\W]','',theremoval)
            titlenospace= re.sub('[\W]','',pre80_lyrics.loc[i,'Title'])
            templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
            templyricURL=templyricURL.lower()

            linktest = requests.get(templyricURL)
            if linktest.status_code == 200:
                templyrichtml=requests.get(templyricURL)
                templyriccontent=html.fromstring(templyrichtml.content)
                templyric=str(templyriccontent.xpath('//*[@id="ldata"]/div[4]/text()'))
                pre80_lyrics.loc[i,'Lyrics']=templyric
                print(4)
            else:
                thetest = re.match('^The ', pre80_lyrics.loc[i,'Artist(s)'])
                if thetest:
                    artnospace= re.sub('[\W]','',pre80_lyrics.loc[i,'Artist(s)'])
                    templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
                    templyricURL=templyricURL.lower()

                    linktest = requests.get(templyricURL)
                    if linktest.status_code == 200:
                        templyrichtml=requests.get(templyricURL)
                        templyriccontent=html.fromstring(templyrichtml.content)
                        templyric=str(templyriccontent.xpath('//*[@id="ldata"]/div[5]/text()'))
                        pre80_lyrics.loc[i,'Lyrics']=templyric
                        print(5)
                    else:
                        if ' and ' in pre80_lyrics.loc[i,'Artist(s)']:
                            andsubremoval=re.sub(' and (?<= and ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
                        else:
                            andsubremoval=pre80_lyrics.loc[i,'Artist(s)']
                        if ' & ' in andsubremoval:
                            ampsubremoval=re.sub(' & (?<= & ).*$','',andsubremoval)
                        else:
                            ampsubremoval=andsubremoval
                        if ' featuring ' in ampsubremoval:
                            featsubremoval=re.sub(' featuring (?<= featuring ).*$','',ampsubremoval)
                        else:
                            featsubremoval=ampsubremoval
                        artnospace= re.sub('[\W]','',featsubremoval)
                else:
                    if ' and ' in pre80_lyrics.loc[i,'Artist(s)']:
                        andsubremoval=re.sub(' and (?<= and ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
                    else:
                        andsubremoval=pre80_lyrics.loc[i,'Artist(s)']
                    if ' & ' in andsubremoval:
                        ampsubremoval=re.sub(' & (?<= & ).*$','',andsubremoval)
                    else:
                        ampsubremoval=andsubremoval
                    if ' featuring ' in ampsubremoval:
                        featsubremoval=re.sub(' featuring (?<= featuring ).*$','',ampsubremoval)
                    else:
                        featsubremoval=ampsubremoval
                    artnospace= re.sub('[\W]','',featsubremoval)
                templyricURL='https://www.lyricsondemand.com/' + firstletter + '/' + artnospace + 'lyrics/' + titlenospace + 'lyrics.html'
                templyricURL=templyricURL.lower()
                templyrichtml=requests.get(templyricURL)
                templyriccontent=html.fromstring(templyrichtml.content)
                templyric=str(templyriccontent.xpath('//*[@id="ldata"]/div[4]/text()'))
                pre80_lyrics.loc[i,'Lyrics']=templyric
                print(6)

print(pre80_lyrics['Lyrics'].value_counts()['[\'\\n\', \'\\n\']'])

#print(pre80_lyrics['Lyrics'].value_counts()['[\'\\n\', \'\\n\\n\']'])

print(pre80_lyrics['Lyrics'].value_counts()['[]'])

print(pre80_lyrics['Lyrics'].value_counts()['[\'\\n\']'])

#figure out which songs still have issues
count=0
for i in pre80_lyrics.index:
    if pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\', \'\\n\']' or pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\']':
        count=count+1
        print(i, ' || ', pre80_lyrics.loc[i,'Lyrics'])

count

#turn lyrics that are just those line breaks into blanks
for i in pre80_lyrics.index:
    if pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\', \'\\n\']' or pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\']':
        pre80_lyrics.loc[i,'Lyrics']='[]'

for i in pre80_lyrics.index:
#for i in range(3925,3926):
    if pre80_lyrics.loc[i,'Lyrics']=='[]':
        if ' and ' in pre80_lyrics.loc[i,'Artist(s)']:
            featart=re.findall('(?<= and ).*$', pre80_lyrics.loc[i,'Artist(s)'])
            featart=str(featart[0])
            andsubremoval=re.sub(' and (?<= and ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
            featadd=andsubremoval+'-feat-'+featart
        elif ' & ' in pre80_lyrics.loc[i,'Artist(s)']:
            featart=re.findall('(?<= & ).*$', pre80_lyrics.loc[i,'Artist(s)'])
            featart=str(featart[0])
            ampsubremoval=re.sub(' & (?<= & ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
            featadd=ampsubremoval+'-feat-'+featart
        elif ' featuring ' in pre80_lyrics.loc[i,'Artist(s)']:
            featart=re.findall('(?<= featuring ).*$', pre80_lyrics.loc[i,'Artist(s)'])
            featart=str(featart[0])
            featsubremoval=re.sub(' featuring (?<= featuring ).*$','',pre80_lyrics.loc[i,'Artist(s)'])
            featadd=featsubremoval+'-feat-'+featart
        else:
            featadd=pre80_lyrics.loc[i,'Artist(s)']
        artnospace=featadd.replace(',','')
        artnospace= re.sub('[\W]','-',artnospace)
        titlenospace=re.sub('^"','',pre80_lyrics.loc[i,'Title'])
        titlenospace=titlenospace.replace(',','')
        titlenospace= re.sub('[\W]','-',titlenospace)
        templyricURL='https://www.songlyrics.com/' + artnospace + '/' + titlenospace + 'lyrics/'
        templyricURL=templyricURL.lower()

        linktest = requests.get(templyricURL)
        if linktest.status_code == 200:
            templyrichtml=requests.get(templyricURL)
            templyriccontent=html.fromstring(templyrichtml.content)
            templyric=str(templyriccontent.xpath('//*[@id="songLyricsDiv"]/text()'))
            pre80_lyrics.loc[i,'Lyrics']=templyric

#print(pre80_lyrics['Lyrics'].value_counts()['[\'\\n\', \'\\n\']'])

#print(pre80_lyrics['Lyrics'].value_counts()['[\'\\n\', \'\\n\\n\']'])

print(pre80_lyrics['Lyrics'].value_counts()['[]'])

print(pre80_lyrics['Lyrics'].value_counts()['[\'\\n\', \'\\n\', \'\\n\\n\']'])

pre80_lyrics.loc[784,'Lyrics']

for i in pre80_lyrics.index:
    if pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\', \'\\n\', \'\\n\\n\']':
        print(i, ' || ', pre80_lyrics.loc[i,'Lyrics'])

for i in pre80_lyrics.index:
    if pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\', \'\\n\', \'\\n\\n\']':
        pre80_lyrics.loc[i,'Lyrics']='[]'

for i in pre80_lyrics.index:
    if pre80_lyrics.loc[i,'Lyrics']=='[\'\\n\', \'\\n\', \'\\n\\n\']':
        print(i, ' || ', pre80_lyrics.loc[i,'Lyrics'])

#print(pre80_lyrics['Lyrics'].value_counts()['[\'\\n\']'])

#print(pre80_lyrics['Lyrics'].value_counts()['[\'We do not have the lyrics for\']'])

#82 songs with no lyrics, definitely good enough to continue.

pre80_lyrics.to_csv(f'{default_directory}/datafiles/pre80_lyrics.csv',index=False)

pre80_lyrics.loc[1905,'Lyrics']

#check breakdown of how many songs are in each
pre80_pivot=pre80_lyrics.pivot_table(columns=['Year'], aggfunc ='size')
pre80_pivot

#make a copy of song list as is before editting
pre80ORG=pre80_lyrics.copy(deep=True)
pre80ORG.to_csv(f'{default_directory}/datafiles/pre80ORG.csv',index=False)

pre80Lyrics=pre80ORG[pre80ORG['Lyrics'] != '[]']
pre80Lyrics=pre80Lyrics.reset_index(drop=True)

pre80Lyrics.to_csv(f'{default_directory}/datafiles/pre80Lyrics.csv',index=False)

#check breakdown again of how many songs are in each
pre80_pivot=pre80Lyrics.pivot_table(columns=['Year'], aggfunc ='size')
pre80_pivot

pre80Lyrics

songArtDF

#remove songs from post 1980 list of songs that had no lyrics
post1980CleanUp=songArtDF.copy(deep=True)
post1980CleanUp=post1980CleanUp[post1980CleanUp['Lyrics'] != '[]']
post1980CleanUp=post1980CleanUp.reset_index(drop=True)
post1980CleanUp

#remove the 'Before2000' column since this excersize is not going to look at that classification.
post1980CleanUp=post1980CleanUp.drop(columns=['Before2000'])
post1980CleanUp

#now combine pre and post 1980 lyrics into one dataframe
tempDF=[pre80Lyrics, post1980CleanUp]
fullLyrList=pd.concat(tempDF)
fullLyrList=fullLyrList.reset_index(drop=True)
fullLyrList

#now create a column that categorizes what decade it is from
for i in fullLyrList.index:
    if fullLyrList.loc[i,'Year'] >= 1950 and fullLyrList.loc[i,'Year'] < 1960:
        fullLyrList.loc[i,'Decade']='1950s'
    elif fullLyrList.loc[i,'Year'] >= 1960 and fullLyrList.loc[i,'Year'] < 1970:
        fullLyrList.loc[i,'Decade']='1960s'
    elif fullLyrList.loc[i,'Year'] >= 1970 and fullLyrList.loc[i,'Year'] < 1980:
        fullLyrList.loc[i,'Decade']='1970s'
    elif fullLyrList.loc[i,'Year'] >= 1980 and fullLyrList.loc[i,'Year'] < 1990:
        fullLyrList.loc[i,'Decade']='1980s'
    elif fullLyrList.loc[i,'Year'] >= 1990 and fullLyrList.loc[i,'Year'] < 2000:
        fullLyrList.loc[i,'Decade']='1990s'
    elif fullLyrList.loc[i,'Year'] >= 2000 and fullLyrList.loc[i,'Year'] < 2010:
        fullLyrList.loc[i,'Decade']='2000s'
    elif fullLyrList.loc[i,'Year'] >= 2010 and fullLyrList.loc[i,'Year'] < 2020:
        fullLyrList.loc[i,'Decade']='2010s'

fullLyrList

decade_pivot=fullLyrList.pivot_table(columns=['Decade'], aggfunc ='size')
decade_pivot

fullLyrList.to_csv(f'{default_directory}/datafiles/fullLyrList.csv',index=False)

################################################################################
### Initial Data Cleanup #######################################################
################################################################################

#now perform Basic data CleanUp that isn't specific to any type of feature analysis
fullLyrBCU=fullLyrList.copy(deep=True)

#remove line breaks
for i in fullLyrBCU.index:
    fullLyrBCU.loc[i,'Lyrics']=fullLyrBCU.loc[i,'Lyrics'].replace('\\n','')
    fullLyrBCU.loc[i,'Lyrics']=fullLyrBCU.loc[i,'Lyrics'].replace('\\r','')
    fullLyrBCU.loc[i,'Lyrics']=fullLyrBCU.loc[i,'Lyrics'].replace('\\','')

fullLyrBCU

#remove contractions
for i in fullLyrBCU.index:
    fullLyrBCU.loc[i,'Lyrics']=contractions.fix(fullLyrBCU.loc[i,'Lyrics'])

fullLyrBCU

#now create a song ID that is just it's current index, which can then be easily used later to merge together different features to perform Naive Bayes
for i in fullLyrBCU.index:
    fullLyrBCU.loc[i,'SongID']='S_'+str(i)

fullLyrBCU

fullLyrBCU.to_csv(f'{default_directory}/datafiles/fullLyrBCU.csv',index=False)

################################################################################
### Pre-Freq Distribution Cleanup Section ######################################
################################################################################

#now perform cleanup needed to eventually do frequency distributions
fullLyrFreqCU=fullLyrBCU.copy(deep=True)

#everything lowercase:
for i in fullLyrFreqCU.index:
    fullLyrFreqCU.loc[i,'Lyrics']=fullLyrFreqCU.loc[i,'Lyrics'].lower()

#remove punctuation
for i in  fullLyrFreqCU.index:
    fullLyrFreqCU.loc[i,'Lyrics']=re.sub(r'[^\w\s]','',fullLyrFreqCU.loc[i,'Lyrics'])

fullLyrFreqCU

fullLyrFreqCU.iloc[2458]

fullLyrFreqCU.to_csv(f'{default_directory}/datafiles/fullLyrFreqCU.csv',index=False)

fullLyrFreqCU.info()

################################################################################
### Frequency Distribution Collection Section ##################################
################################################################################

#now create a dataframe that is going to go through the full list of lyrics,
# and for each song tokenize the lyrics, make a frequency distribution,
# then attach the frequency distribution to the songID and the decade for classification

#first some testing

#create code that will take the lyrics from each song and make it into a frequency distribution
for i in range(0,1):
    #tokenize the current song lyrics
    tempWordTok=nltk.word_tokenize(fullLyrFreqCU.loc[i,'Lyrics'])
    #remove stop words
    tempTokStop=[w for w in tempWordTok if not w in stopwords]
    #create the frequency distribution
    tempDist=FreqDist(tempTokStop)

tempDist

tempDist['goodnight']

for h in tempDist:
    print(h, tempDist[h])

#now the code that will turn the frequency distribution into a data frame
LyrFreqDF=pd.DataFrame()
SongID=fullLyrFreqCU.loc[i,'SongID']
freqVals=[l for l in tempDist.values()]
freqCols=[l for l in tempDist.keys()]
TimeForDF=[SongID]
TimeForDF.extend(freqVals)
TempFreqInfo = pd.DataFrame([TimeForDF])
TempFreqInfo=TempFreqInfo.rename(columns={0:'SongID'})
for c in range(0,len(freqCols)):
    TempFreqInfo=TempFreqInfo.rename(columns={c+1:freqCols[c]})
CombFreqInfo=[LyrFreqDF,TempFreqInfo]
LyrFreqDF=pd.concat(CombFreqInfo)

LyrFreqDF

#now combine and test

LyrFreqDF=pd.DataFrame()
for i in range(0,5):
    #grab the song ID
    SongID=fullLyrFreqCU.loc[i,'SongID']
    Decade=fullLyrFreqCU.loc[i,'Decade']
    #tokenize the current song lyrics
    tempWordTok=nltk.word_tokenize(fullLyrFreqCU.loc[i,'Lyrics'])
    #remove stop words
    tempTokStop=[w for w in tempWordTok if not w in stopwords]
    #create the frequency distribution
    tempDist=FreqDist(tempTokStop)

    freqVals=[l for l in tempDist.values()]
    freqCols=[l for l in tempDist.keys()]
    TimeForDF=[SongID,Decade]
    TimeForDF.extend(freqVals)
    TempFreqInfo = pd.DataFrame([TimeForDF])
    TempFreqInfo=TempFreqInfo.rename(columns={0:'SongID'})
    TempFreqInfo=TempFreqInfo.rename(columns={1:'Decade'})
    for c in range(0,len(freqCols)):
        TempFreqInfo=TempFreqInfo.rename(columns={c+2:freqCols[c]})
    CombFreqInfo=[LyrFreqDF,TempFreqInfo]
    LyrFreqDF=pd.concat(CombFreqInfo)

TempFreqInfo

LyrFreqDF

######IMPORTANT NOTE# This code took over 6 hours to run, so once it finished I
# exported the CSV file so that it can just be imported going forward.
# There is code to import it at the top of the page already, the file is LyrFreqDF.

#now run the real thing
'''
LyrFreqDF=pd.DataFrame()
for i in fullLyrFreqCU.index:
    #grab the song ID
    SongID=fullLyrFreqCU.loc[i,'SongID']
    Decade=fullLyrFreqCU.loc[i,'Decade']
    #tokenize the current song lyrics
    tempWordTok=nltk.word_tokenize(fullLyrFreqCU.loc[i,'Lyrics'])
    #remove stop words
    tempTokStop=[w for w in tempWordTok if not w in stopwords]
    #create the frequency distribution
    tempDist=FreqDist(tempTokStop)

    #take the frequency distribution and turn it into a row of a dataframe, then add it back to the original data frame
    freqVals=[l for l in tempDist.values()]
    freqCols=[l for l in tempDist.keys()]
    TimeForDF=[SongID,Decade]
    TimeForDF.extend(freqVals)
    TempFreqInfo = pd.DataFrame([TimeForDF])
    TempFreqInfo=TempFreqInfo.rename(columns={0:'SongID'})
    TempFreqInfo=TempFreqInfo.rename(columns={1:'Decade'})
    for c in range(0,len(freqCols)):
        TempFreqInfo=TempFreqInfo.rename(columns={c+2:freqCols[c]})
    CombFreqInfo=[LyrFreqDF,TempFreqInfo]
    LyrFreqDF=pd.concat(CombFreqInfo)
'''

#make sure to output this when done so that it can just be loaded going forward
#LyrFreqDF=LyrFreqDF.reset_index(drop=True)
#LyrFreqDF=LyrFreqDF.fillna(0)
#LyrFreqDF.to_csv(f'{default_directory}/datafiles/LyrFreqDF.csv',index=False)

######IMPORTANT NOTE# This code also took a long time to run, so once it finished I
# exported the CSV file so that it can just be imported going forward.
# There is code to import it at the top of the page already, the file is LyrFreqDF_2

#Now try a frequency distribution where words are changed to be the same tense, and plural is changed to singular

'''
LyrFreqDF_2=pd.DataFrame()
for i in fullLyrFreqCU.index:
    #grab the song ID
    SongID=fullLyrFreqCU.loc[i,'SongID']
    Decade=fullLyrFreqCU.loc[i,'Decade']
    #tokenize the current song lyrics
    tempWordTok=nltk.word_tokenize(fullLyrFreqCU.loc[i,'Lyrics'])
    #remove stop words
    tempTokStop=[w for w in tempWordTok if not w in stopwords]

    ####change the tense of words:
    tokTenseTags = nltk.pos_tag(tempTokStop)
    tokTenseNew=[]
    for ct in tokTenseTags:
        tagList=re.match(r'V',ct[1]) or re.match(r'JJ',ct[1])
        if tagList:
            tokTenseNew.append(lem.lemmatize(ct[0],'v'))
        else:
            tokTenseNew.append(ct[0])

    #change plural to singular
    for p in range(0,len(tokTenseNew)):
        if wordPlur.singular_noun(tokTenseNew[p]) == False:
            continue
        else:
            tokTenseNew[p]=wordPlur.singular_noun(tokTenseNew[p])

    #create the frequency distribution
    tempDist=FreqDist(tokTenseNew)

    #take the frequency distribution and turn it into a row of a dataframe, then add it back to the original data frame
    freqVals=[l for l in tempDist.values()]
    freqCols=[l for l in tempDist.keys()]
    TimeForDF=[SongID,Decade]
    TimeForDF.extend(freqVals)
    TempFreqInfo = pd.DataFrame([TimeForDF])
    TempFreqInfo=TempFreqInfo.rename(columns={0:'SongID'})
    TempFreqInfo=TempFreqInfo.rename(columns={1:'Decade'})
    for c in range(0,len(freqCols)):
        TempFreqInfo=TempFreqInfo.rename(columns={c+2:freqCols[c]})
    CombFreqInfo=[LyrFreqDF_2,TempFreqInfo]
    LyrFreqDF_2=pd.concat(CombFreqInfo)
'''

#LyrFreqDF_2=LyrFreqDF_2.reset_index(drop=True)
#LyrFreqDF_2=LyrFreqDF_2.fillna(0)
#LyrFreqDF_2.to_csv(f'{default_directory}/datafiles/LyrFreqDF_2.csv',index=False)

################################################################################
### Sentiment Analysis Collection Section ######################################
################################################################################

#first create a sent analysis data frame
LyrSentDF=pd.DataFrame()
for i in fullLyrFreqCU.index:
    #grab the song ID
    SongID=fullLyrFreqCU.loc[i,'SongID']
    Decade=fullLyrFreqCU.loc[i,'Decade']
    #tokenize the current song lyrics
    tempWordTok=nltk.word_tokenize(fullLyrFreqCU.loc[i,'Lyrics'])
    #remove stop words
    tempTokStop=[w for w in tempWordTok if not w in stopwords]
    #sent analysis needs to be peformed against string, not a list, so reverting the words back to a single string
    tempSTR=''
    for h in tempTokStop:
        tempSTR=tempSTR + ' ' + h
    #perform sent analysis on the lyrics
    tempSent=sid.polarity_scores(tempSTR)

    #take the sent analysis and turn it into a row of a dataframe, then add it back to the original data frame
    Neg=tempSent['neg']
    Pos=tempSent['pos']
    Neu=tempSent['neu']
    Comp=tempSent['compound']
    TimeForDF=[SongID, Decade, Neg, Pos, Neu, Comp]
    TempSentInfo = pd.DataFrame([TimeForDF],columns=('SongID','Decade','Negative', 'Positive','Neutral','Compound'))
    CombSentInfo=[LyrSentDF,TempSentInfo]
    LyrSentDF=pd.concat(CombSentInfo)

LyrSentDF=LyrSentDF.reset_index(drop=True)
LyrSentDF=LyrSentDF.fillna(0)

LyrSentDF.to_csv(f'{default_directory}/datafiles/LyrSentDF.csv',index=False)

LyrSentDF

sent_pivot=LyrSentDF.pivot_table(columns=['Decade'], aggfunc ='mean')
sent_pivot

################################################################################
### Empath Analysis Section ####################################################
################################################################################

#now explore the empath analysis, first test it to see what the output is like
#for i in fullLyrFreqCU.index:
for i in range(0,1):
    #grab the song ID
    SongID=fullLyrFreqCU.loc[i,'SongID']
    Decade=fullLyrFreqCU.loc[i,'Decade']
    #tokenize the current song lyrics
    tempWordTok=nltk.word_tokenize(fullLyrFreqCU.loc[i,'Lyrics'])
    #remove stop words
    tempTokStop=[w for w in tempWordTok if not w in stopwords]
    #perform emp analysis
    tempWordEmp=lexicon.analyze(tempTokStop,normalize=True)

print(len(tempWordEmp))
tempWordEmp

tempWordEmp.values()

tempWordEmp.keys()

print(lexicon.cats.keys())
len(lexicon.cats.keys())

freqCols=[l for l in lexicon.cats.keys()]
freqCols

#now test adding in the data frame creation
LyrEmpDF=pd.DataFrame()
#for i in fullLyrFreqCU.index:
for i in range(0,1):
    #grab the song ID
    SongID=fullLyrFreqCU.loc[i,'SongID']
    Decade=fullLyrFreqCU.loc[i,'Decade']
    #tokenize the current song lyrics
    tempWordTok=nltk.word_tokenize(fullLyrFreqCU.loc[i,'Lyrics'])
    #remove stop words
    tempTokStop=[w for w in tempWordTok if not w in stopwords]
    #perform emp analysis
    tempWordEmp=lexicon.analyze(tempTokStop,normalize=True)

    #take the Emp scores and turn them into a row of a dataframe, then add it back to the original data frame
    freqVals=[l for l in tempWordEmp.values()]
    TimeForDF=[SongID,Decade]
    TimeForDF.extend(freqVals)
    TempFreqInfo = pd.DataFrame([TimeForDF])
    TempFreqInfo=TempFreqInfo.rename(columns={0:'SongID'})
    TempFreqInfo=TempFreqInfo.rename(columns={1:'Decade'})
    for c in range(0,len(freqCols)):
        TempFreqInfo=TempFreqInfo.rename(columns={c+2:freqCols[c]})
    CombFreqInfo=[LyrEmpDF,TempFreqInfo]
    LyrEmpDF=pd.concat(CombFreqInfo)

LyrEmpDF

#now performing the full analysis

freqCols=[l for l in lexicon.cats.keys()]

LyrEmpDF=pd.DataFrame()
for i in fullLyrFreqCU.index:
     #grab the song ID
    SongID=fullLyrFreqCU.loc[i,'SongID']
    Decade=fullLyrFreqCU.loc[i,'Decade']
    #tokenize the current song lyrics
    tempWordTok=nltk.word_tokenize(fullLyrFreqCU.loc[i,'Lyrics'])
    #remove stop words
    tempTokStop=[w for w in tempWordTok if not w in stopwords]
    #perform emp analysis
    tempWordEmp=lexicon.analyze(tempTokStop,normalize=True)

    #take the Emp scores and turn them into a row of a dataframe, then add it back to the original data frame
    freqVals=[l for l in tempWordEmp.values()]
    TimeForDF=[SongID,Decade]
    TimeForDF.extend(freqVals)
    TempFreqInfo = pd.DataFrame([TimeForDF])
    TempFreqInfo=TempFreqInfo.rename(columns={0:'SongID'})
    TempFreqInfo=TempFreqInfo.rename(columns={1:'Decade'})
    for c in range(0,len(freqCols)):
        TempFreqInfo=TempFreqInfo.rename(columns={c+2:freqCols[c]})
    CombFreqInfo=[LyrEmpDF,TempFreqInfo]
    LyrEmpDF=pd.concat(CombFreqInfo)

LyrEmpDF=LyrEmpDF.reset_index(drop=True)
LyrEmpDF=LyrEmpDF.fillna(0)

len(fullLyrFreqCU)

LyrEmpDF.to_csv(f'{default_directory}/datafiles/LyrEmpDF.csv',index=False)

LyrEmpDF

####Remove any columns that have a sum of 0, meaning none of the lyrics had a word that fell under that columns category
for i in LyrEmpDF.columns:
    if i != 'SongID' and i != 'Decade':
        if sum(LyrEmpDF[i])==0:
            print(i)

empSumCheck=[]
for i in LyrEmpDF.columns:
    if i != 'SongID' and i != 'Decade':
        empSumCheck.append(sum(LyrEmpDF[i]))

print(len(empSumCheck))
empSumCheck

################################################################################
### LIWC Analysis Section ######################################################
################################################################################

print(liwc.search('happy'))

liwc.categories

liwc.lexicon

liwc.parse('this is a test of how this works.')

liwc.parse('this is a test of how this works.'.split(' '))

testSentliwc='this is a test of how this works.'
tempWordTok=nltk.word_tokenize(testSentliwc)
for i in tempWordTok:
    print(i, ' || ', liwc.parse(i))

testSentliwc='this is a test of how this works.'
tempWordTok=nltk.word_tokenize(testSentliwc)
for i in tempWordTok:
    print(i, ' || ', liwc.parse(i.split(' ')))

###It appears that when the words are tokenized AND .split() is added to the end, we get the most rebust results
testSentliwc='this is a test of how this works'
tempWordTok=nltk.word_tokenize(testSentliwc)
for i in tempWordTok:
    print(i, ' || ', liwc.parse(i.split()))
    testLiwcList=liwc.parse(i.split())

print(testLiwcList.keys())
print(testLiwcList.values())
print(testLiwcList['work@Work'])
for i in testLiwcList:
    print(i)

###now test how it will run against the data
LyrLiwcDF=pd.DataFrame()
for i in fullLyrFreqCU.index:
#for i in range(0,1):
     #grab the song ID
    SongID=fullLyrFreqCU.loc[i,'SongID']
    Decade=fullLyrFreqCU.loc[i,'Decade']
    #tokenize the current song lyrics
    tempWordTok=nltk.word_tokenize(fullLyrFreqCU.loc[i,'Lyrics'])
    #remove stop words
    tempTokStop=[w for w in tempWordTok if not w in stopwords]
    #perform liwc analysis
    tempLiwcDict={}
    for h in tempTokStop:
        tempLiwc=liwc.parse(h.split())
        for d in tempLiwc:
            if d in tempLiwcDict:
                tempLiwcDict[d]=tempLiwcDict[d]+1
            else:
                tempLiwcDict[d]=1

    #take the liwc scores and turn them into a row of a dataframe, then add it back to the original data frame
    liwcVals=[l for l in tempLiwcDict.values()]
    liwcCols=[l for l in tempLiwcDict.keys()]
    TimeForDF=[SongID,Decade]
    TimeForDF.extend(liwcVals)
    TempLiwcInfo = pd.DataFrame([TimeForDF])
    TempLiwcInfo=TempLiwcInfo.rename(columns={0:'SongID'})
    TempLiwcInfo=TempLiwcInfo.rename(columns={1:'Decade'})
    for c in range(0,len(liwcCols)):
        TempLiwcInfo=TempLiwcInfo.rename(columns={c+2:liwcCols[c]})
    CombLiwcInfo=[LyrLiwcDF,TempLiwcInfo]
    LyrLiwcDF=pd.concat(CombLiwcInfo)

LyrLiwcDF=LyrLiwcDF.reset_index(drop=True)
LyrLiwcDF=LyrLiwcDF.fillna(0)

LyrLiwcDF.to_csv(f'{default_directory}/datafiles/LyrLiwcDF.csv',index=False)

LyrLiwcDF

################################################################################
### POS Tagging Data Collection Section ########################################
################################################################################

fullLyrBCU

POScleanUp=fullLyrBCU.copy(deep=True)

POScleanUp.loc[2235,'Lyrics']

####To get this split into sentences, need to replace commas in the current lyric states with a period. Accomplish this with a series of sub regex statements.
for i in POScleanUp.index:
    POScleanUp.loc[i,'Lyrics']=re.sub('\',','.',POScleanUp.loc[i,'Lyrics'])

for i in POScleanUp.index:
    POScleanUp.loc[i,'Lyrics']=re.sub('\",','.',POScleanUp.loc[i,'Lyrics'])

for i in POScleanUp.index:
    POScleanUp.loc[i,'Lyrics']=re.sub('\']','.',POScleanUp.loc[i,'Lyrics'])

for i in POScleanUp.index:
    POScleanUp.loc[i,'Lyrics']=re.sub('\"]','.',POScleanUp.loc[i,'Lyrics'])

POScleanUp.to_csv(f'{default_directory}/datafiles/POScleanUp.csv',index=False)

POScleanUp

#now build a data frame that contains the number of lines per song, number of words per line, and number of characters per line in each song
LyrPOSDF=pd.DataFrame()
for i in POScleanUp.index:
    #grab the song ID and decade
    SongID=POScleanUp.loc[i,'SongID']
    Decade=POScleanUp.loc[i,'Decade']
    #split the lyrics into sentences
    tempSplitLyr=nltk.sent_tokenize(POScleanUp.loc[i,'Lyrics'])
    #remove punctuation
    tempSplitLyr=[re.sub(r'[^\w\s]','',p) for p in tempSplitLyr]
    #remove lines that only contained punctuation and are now empty
    tempSplitList=[h for h in tempSplitLyr if h != '']
    #calculate lines per song
    LinePerSong=len(tempSplitList)
    #calculate characters per line
    CharTotal = sum(len(s) for s in tempSplitList)
    CharPerLine=CharTotal/len(tempSplitList)
    #calc words per line
    WordTotal = sum([len(w) for w in [nltk.word_tokenize(s) for s in tempSplitList]])
    WordPerLine=WordTotal/len(tempSplitList)
    #get the pos tags
    tempPOSDict={}
    for t in tempSplitList:
        t=t.lower()
        POSToks = nltk.word_tokenize(t)
        POSTags = nltk.pos_tag(POSToks)
        tempPOSCt=Counter(tag for _, tag in POSTags)
        for d in tempPOSCt:
            if d in tempPOSDict:
                tempPOSDict[d]=tempPOSDict[d]+tempPOSCt[d]
            else:
                tempPOSDict[d]=tempPOSCt[d]

    #now build the row in the DF
    POSVals=[l for l in tempPOSDict.values()]
    POSCols=[l for l in tempPOSDict.keys()]
    TimeForDF=[SongID, Decade,LinePerSong, CharPerLine, WordPerLine]
    TimeForDF.extend(POSVals)
    TempPOSInfo = pd.DataFrame([TimeForDF])
    TempPOSInfo=TempPOSInfo.rename(columns={0:'SongID'})
    TempPOSInfo=TempPOSInfo.rename(columns={1:'Decade'})
    TempPOSInfo=TempPOSInfo.rename(columns={2:'LinePerSong'})
    TempPOSInfo=TempPOSInfo.rename(columns={3:'CharPerLine'})
    TempPOSInfo=TempPOSInfo.rename(columns={4:'WordPerLine'})
    for c in range(0,len(POSCols)):
        TempPOSInfo=TempPOSInfo.rename(columns={c+5:POSCols[c]})
    CombPOSInfo=[LyrPOSDF,TempPOSInfo]
    LyrPOSDF=pd.concat(CombPOSInfo)

LyrPOSDF=LyrPOSDF.reset_index(drop=True)

LyrPOSDF=LyrPOSDF.fillna(0)

LyrPOSDF.to_csv(f'{default_directory}/datafiles/LyrPOSDF.csv',index=False)

LyrPOSDF

#now repeat but use nltk stop words before taking any counts
LyrPOSDF_NLTK=pd.DataFrame()
ErrorRecord=[]
for i in POScleanUp.index:
    #grab the song ID and decade
    SongID=POScleanUp.loc[i,'SongID']
    Decade=POScleanUp.loc[i,'Decade']
    #split the lyrics into sentences
    tempSplitLyr=nltk.sent_tokenize(POScleanUp.loc[i,'Lyrics'])
    #remove punctuation
    tempSplitLyr=[re.sub(r'[^\w\s]','',p) for p in tempSplitLyr]
    #remove lines that only contained punctuation and are now empty
    tempSplitList=[h for h in tempSplitLyr if h != '']
    #tokenize the words in the lines
    tokSplitList=[]
    for s in tempSplitList:
        tokSplitList.append(nltk.word_tokenize(s))
    #remove the stop words
    StSplitList=[]
    for sw in tokSplitList:
        StSplitList.append([w.lower() for w in sw if not w in stopwords])
    #remove lines that may have only contained stop words and are now empty
    StSplitList=[h for h in StSplitList if h != '']
    #calculate lines per song
    LinePerSong=len(StSplitList)
    if LinePerSong==0:
        CharPerLine=0
        WordPerLine=0
        print('Index with error:',i,' || Before Lyrics:', POScleanUp.loc[i,'Lyrics'], ' || Edited Lyrics:',  StSplitList)
        ErrorRecord.append(i)
    else:
    #calculate characters per line
        tempStString=''
        for c in StSplitList:
            for w in range(0,len(c)):
                tempStString=tempStString+' '+c[w]
        CharPerLine=len(tempStString)/len(StSplitList)
    #calc words per line
        WordTotal = sum([len(w) for w in StSplitList])
        WordPerLine=WordTotal/len(StSplitList)
    #get the pos tags
        tempPOSDict={}
        for t in StSplitList:
            POSTags = nltk.pos_tag(t)
            tempPOSCt=Counter(tag for _, tag in POSTags)
            for d in tempPOSCt:
                if d in tempPOSDict:
                    tempPOSDict[d]=tempPOSDict[d]+tempPOSCt[d]
                else:
                    tempPOSDict[d]=tempPOSCt[d]

    #now build the row in the DF
    POSVals=[l for l in tempPOSDict.values()]
    POSCols=[l for l in tempPOSDict.keys()]
    TimeForDF=[SongID, Decade,LinePerSong, CharPerLine, WordPerLine]
    TimeForDF.extend(POSVals)
    TempPOSInfo = pd.DataFrame([TimeForDF])
    TempPOSInfo=TempPOSInfo.rename(columns={0:'SongID'})
    TempPOSInfo=TempPOSInfo.rename(columns={1:'Decade'})
    TempPOSInfo=TempPOSInfo.rename(columns={2:'LinePerSong'})
    TempPOSInfo=TempPOSInfo.rename(columns={3:'CharPerLine'})
    TempPOSInfo=TempPOSInfo.rename(columns={4:'WordPerLine'})
    for c in range(0,len(POSCols)):
        TempPOSInfo=TempPOSInfo.rename(columns={c+5:POSCols[c]})
    CombPOSInfo=[LyrPOSDF_NLTK,TempPOSInfo]
    LyrPOSDF_NLTK=pd.concat(CombPOSInfo)

LyrPOSDF_NLTK=LyrPOSDF_NLTK.reset_index(drop=True)

LyrPOSDF_NLTK=LyrPOSDF_NLTK.fillna(0)

LyrPOSDF_NLTK.to_csv(f'{default_directory}/datafiles/LyrPOSDF_NLTK.csv',index=False)

LyrPOSDF_NLTK

#now use spacey tokenization and spacey stop words
LyrPOSDF_SPC=pd.DataFrame()
ErrorRecord=[]
for i in POScleanUp.index:
    #grab the song ID and decade
    SongID=POScleanUp.loc[i,'SongID']
    Decade=POScleanUp.loc[i,'Decade']
    #split the lyrics into sentences, which in this spacey technique requires word tokens first
    tempSplitLyr=[sent for sent in sp(POScleanUp.loc[i,'Lyrics']).sents]
    #remove punctuation
    tempSplitLyr=[re.sub(r'[^\w\s]','',str(p)) for p in tempSplitLyr]
    #remove lines that only contained punctuation and are now empty
    tempSplitList=[h for h in tempSplitLyr if h != '']
    #tokenize the words in the lines
    tokSplitList=[]
    for s in tempSplitList:
        tokSplitList.append(sp(s))
    #remove the stop words
    StSplitList=[]
    for sw in tokSplitList:
        StSplitList.append([str(w).lower() for w in sw if not str(w).lower() in all_stopwords])
    #remove empty entries:
    for q in range(0,len(StSplitList)):
        StSplitList[q]=' '.join(StSplitList[q]).split()
    #remove lines that may have only contained stop words and are now empty
    StSplitList=[h for h in StSplitList if h != '']
    #remove words that may just be blank brackets now
    StSplitList=[h for h in StSplitList if h != []]
    #calculate lines per song
    LinePerSong=len(StSplitList)
    if LinePerSong==0:
        CharPerLine=0
        WordPerLine=0
        print('Index with error:',i,' || Before Lyrics:', POScleanUp.loc[i,'Lyrics'], ' || Edited Lyrics:',  StSplitList)
        ErrorRecord.append(i)
    else:
    #calculate characters per line
        tempStString=''
        for c in StSplitList:
            for w in range(0,len(c)):
                tempStString=tempStString+' '+c[w]
        CharPerLine=len(tempStString)/len(StSplitList)
    #calc words per line
        WordTotal = sum([len(w) for w in StSplitList])
        WordPerLine=WordTotal/len(StSplitList)
    #get the pos tags
        tempPOSDict={}
        for t in StSplitList:
            POSTags = nltk.pos_tag(t)
            tempPOSCt=Counter(tag for _, tag in POSTags)
            for d in tempPOSCt:
                if d in tempPOSDict:
                    tempPOSDict[d]=tempPOSDict[d]+tempPOSCt[d]
                else:
                    tempPOSDict[d]=tempPOSCt[d]

    #now build the row in the DF
    POSVals=[l for l in tempPOSDict.values()]
    POSCols=[l for l in tempPOSDict.keys()]
    TimeForDF=[SongID, Decade,LinePerSong, CharPerLine, WordPerLine]
    TimeForDF.extend(POSVals)
    TempPOSInfo = pd.DataFrame([TimeForDF])
    TempPOSInfo=TempPOSInfo.rename(columns={0:'SongID'})
    TempPOSInfo=TempPOSInfo.rename(columns={1:'Decade'})
    TempPOSInfo=TempPOSInfo.rename(columns={2:'LinePerSong'})
    TempPOSInfo=TempPOSInfo.rename(columns={3:'CharPerLine'})
    TempPOSInfo=TempPOSInfo.rename(columns={4:'WordPerLine'})
    for c in range(0,len(POSCols)):
        TempPOSInfo=TempPOSInfo.rename(columns={c+5:POSCols[c]})
    CombPOSInfo=[LyrPOSDF_SPC,TempPOSInfo]
    LyrPOSDF_SPC=pd.concat(CombPOSInfo)

i

POScleanUp.loc[2235,'Title']

LinePerSong

LyrPOSDF_SPC=LyrPOSDF_SPC.reset_index(drop=True)

LyrPOSDF_SPC=LyrPOSDF_SPC.fillna(0)

LyrPOSDF_SPC.to_csv(f'{default_directory}/datafiles/LyrPOSDF_SPC.csv',index=False)

LyrPOSDF_SPC

#now finally use gensim and regex
LyrPOSDF_GM=pd.DataFrame()
ErrorRecord=[]
for i in POScleanUp.index:
    #grab the song ID and decade
    SongID=POScleanUp.loc[i,'SongID']
    Decade=POScleanUp.loc[i,'Decade']
    #split the lyrics into sentences using regex split this time
    tempSplitLyr=re.compile('[.!?] ').split(POScleanUp.loc[i,'Lyrics'])
    #remove punctuation
    tempSplitLyr=[re.sub(r'[^\w\s]','',p) for p in tempSplitLyr]
    #remove lines that only contained punctuation and are now empty
    tempSplitList=[h for h in tempSplitLyr if h != '']
    #tokenize the words in the lines
    tokSplitList=[]
    for s in tempSplitList:
        tokSplitList.append(list(tokenize(s)))
    #remove the stop words
    StSplitList=[]
    for sw in tokSplitList:
        StSplitList.append([remove_stopwords(w.lower()) for w in sw])
    #remove empty entries:
    for q in range(0,len(StSplitList)):
        StSplitList[q]=' '.join(StSplitList[q]).split()
    #remove lines that may have only contained stop words and are now empty
    StSplitList=[h for h in StSplitList if h != '']
    #remove words that may just be blank brackets now
    StSplitList=[h for h in StSplitList if h != []]
    #calculate lines per song
    LinePerSong=len(StSplitList)
    if LinePerSong==0:
        CharPerLine=0
        WordPerLine=0
        print('Index with error:',i,' || Before Lyrics:', POScleanUp.loc[i,'Lyrics'], ' || Edited Lyrics:',  StSplitList)
        ErrorRecord.append(i)
    else:
    #calculate characters per line
        tempStString=''
        for c in StSplitList:
            for w in range(0,len(c)):
                tempStString=tempStString+' '+c[w]
        CharPerLine=len(tempStString)/len(StSplitList)
    #calc words per line
        WordTotal = sum([len(w) for w in StSplitList])
        WordPerLine=WordTotal/len(StSplitList)
    #get the pos tags
        tempPOSDict={}
        for t in StSplitList:
            POSTags = nltk.pos_tag(t)
            tempPOSCt=Counter(tag for _, tag in POSTags)
            for d in tempPOSCt:
                if d in tempPOSDict:
                    tempPOSDict[d]=tempPOSDict[d]+tempPOSCt[d]
                else:
                    tempPOSDict[d]=tempPOSCt[d]

    #now build the row in the DF
    POSVals=[l for l in tempPOSDict.values()]
    POSCols=[l for l in tempPOSDict.keys()]
    TimeForDF=[SongID, Decade,LinePerSong, CharPerLine, WordPerLine]
    TimeForDF.extend(POSVals)
    TempPOSInfo = pd.DataFrame([TimeForDF])
    TempPOSInfo=TempPOSInfo.rename(columns={0:'SongID'})
    TempPOSInfo=TempPOSInfo.rename(columns={1:'Decade'})
    TempPOSInfo=TempPOSInfo.rename(columns={2:'LinePerSong'})
    TempPOSInfo=TempPOSInfo.rename(columns={3:'CharPerLine'})
    TempPOSInfo=TempPOSInfo.rename(columns={4:'WordPerLine'})
    for c in range(0,len(POSCols)):
        TempPOSInfo=TempPOSInfo.rename(columns={c+5:POSCols[c]})
    CombPOSInfo=[LyrPOSDF_GM,TempPOSInfo]
    LyrPOSDF_GM=pd.concat(CombPOSInfo)

LyrPOSDF_GM=LyrPOSDF_GM.reset_index(drop=True)

LyrPOSDF_GM=LyrPOSDF_GM.fillna(0)

LyrPOSDF_GM.to_csv(f'{default_directory}/datafiles/LyrPOSDF_GM.csv',index=False)

LyrPOSDF_GM

################################################################################
### Begin Feature Set and Modeling Work : Freq Dist ############################
################################################################################

#LyrFreqDF || original freq dist DF

#LyrFreqDF_2 || freq dist with words changed to same tense and singularized

#IMPORTANT NOTE# This code once again took a while to run, so once it finished I
# exported the JSON so that it can just be imported going forward.
# There is code to import it at the top of the page already, the file is feature_set_freqDist

'''
#collect feature set from original freq dist
feature_set_freqDist=[]
for i in LyrFreqDF.index:
    features_dict={}
    features_item=()
    for h in range(2,len(LyrFreqDF.columns)):
        features_dict[LyrFreqDF.columns[h]]=LyrFreqDF.loc[i,LyrFreqDF.columns[h]]
    features_item=(features_dict,LyrFreqDF.loc[i,'Decade'])
    feature_set_freqDist.append(features_item)
    '''

'''Exporting it also took a while.
#since it takes so long to create this feature set, export it and then import it
with open(f'{default_directory}/json_files/feature_set_freqDist.json', 'w') as file:
    json.dump(feature_set_freqDist, file)
    '''

#now create the test and train set
random.shuffle(feature_set_freqDist)
cutoff=.80
split=round(len(feature_set_freqDist)*cutoff)
train_set=feature_set_freqDist[:split]
test_set=feature_set_freqDist[split:]

#create the model
freqDist_train_model=nltk.NaiveBayesClassifier.train(train_set)

#test for accuracy
nltk.classify.accuracy(freqDist_train_model, test_set)

#IMPORTANT NOTE# This code once again took a while to run, so once it finished I
# exported the JSON so that it can just be imported going forward.
# There is code to import it at the top of the page already, the file is feature_set_freqDist_2
'''
#collect feature set from original freq dist with tense changed and singularized
feature_set_freqDist_2=[]
for i in LyrFreqDF_2.index:
    features_dict={}
    features_item=()
    for h in range(2,len(LyrFreqDF_2.columns)):
        features_dict[LyrFreqDF_2.columns[h]]=LyrFreqDF_2.loc[i,LyrFreqDF_2.columns[h]]
    features_item=(features_dict,LyrFreqDF_2.loc[i,'Decade'])
    feature_set_freqDist_2.append(features_item)
    '''

'''Exporting it also took a while.
#since it takes so long to create this feature set, export it and then import it
with open(f'{default_directory}/json_files/feature_set_freqDist_2.json', 'w') as file:
    json.dump(feature_set_freqDist_2, file)
    '''

#now create the test and train set
random.shuffle(feature_set_freqDist_2)
cutoff=.80
split=round(len(feature_set_freqDist_2)*cutoff)
train_set=feature_set_freqDist_2[:split]
test_set=feature_set_freqDist_2[split:]

#create the model
freqDist_2_train_model=nltk.NaiveBayesClassifier.train(train_set)

#test for accuracy
nltk.classify.accuracy(freqDist_2_train_model, test_set)

## cross-validation ##
# this function takes the number of folds, the feature sets
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the accuracy for each fold and the average accuracy at the end
def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

cross_validation_accuracy(10,feature_set_freqDist_2)

cross_validation_accuracy(10,feature_set_freqDist)

################################################################################
### Now look at confusion matrix, precision, recall, F1 ########################
################################################################################

#####first looking at original freq distribution#####
#now create the test and train set
random.shuffle(feature_set_freqDist)
cutoff=.80
split=round(len(feature_set_freqDist)*cutoff)
train_set=feature_set_freqDist[:split]
test_set=feature_set_freqDist[split:]

goldlist = []
predictedlist = []
for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(freqDist_train_model.classify(features))

# look at the first 30 examples
print(goldlist[:30])
print(predictedlist[:30])

cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))

# or show the results as percentages
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

# call the function with our data
eval_measures(goldlist, predictedlist)

#####now looking at 2nd, reduced, freq dist#####
#now create the test and train set
#now create the test and train set
random.shuffle(feature_set_freqDist_2)
cutoff=.80
split=round(len(feature_set_freqDist_2)*cutoff)
train_set=feature_set_freqDist_2[:split]
test_set=feature_set_freqDist_2[split:]

goldlist = []
predictedlist = []
for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(freqDist_2_train_model.classify(features))

# look at the first 30 examples
print(goldlist[:30])
print(predictedlist[:30])

cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))

# or show the results as percentages
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

# call the function with our data
eval_measures(goldlist, predictedlist)

################################################################################
### Use sklearn and top n predictors instead ###################################
################################################################################

X_train, X_test, y_train, y_test = train_test_split(LyrFreqDF_2.iloc[:,2:], LyrFreqDF_2['Decade'])

SKNB_FreqDF_2=MultinomialNB()
SKNB_FreqDF_2.fit(X_train,y_train)

SKNB_FreqDF_Score=SKNB_FreqDF_2.score(X_test,y_test)
SKNB_FreqDF_Score

cm_sk=confusion_matrix(LyrFreqDF_2['Decade'],SKNB_FreqDF_2.predict(LyrFreqDF_2.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SKNB_FreqDF_2.classes_)
cm_sk_dsp.plot()
plt.show()

cm_sk_n=confusion_matrix(LyrFreqDF_2['Decade'],SKNB_FreqDF_2.predict(LyrFreqDF_2.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SKNB_FreqDF_2.classes_)
cm_sk_dsp_n.plot()
plt.show()

SKNB_estimatorFreqDF_2=SKNB_FreqDF_2.predict_proba(X_test)
SKNB_estimatorFreqDF_2[0]

SKNB_topK=top_k_accuracy_score(y_test,SKNB_estimatorFreqDF_2,k=3)

SKNB_topK

################################################################################
### Now try SVMs instead #######################################################
################################################################################

svm_test=svm.SVC(decision_function_shape='ovo')

svm_test2=svm.SVC(decision_function_shape='ovr')

svm_test_model=svm_test.fit(X_train,y_train)

svm_test_model2=svm_test2.fit(X_train,y_train)

svm_test_model.score(X_test,y_test)

svm_test_model2.score(X_test,y_test)

SVM_CM=svm_test.fit(LyrFreqDF_2.iloc[:,2:],LyrFreqDF_2['Decade'])
cm_sk=confusion_matrix(LyrFreqDF_2['Decade'],SVM_CM.predict(LyrFreqDF_2.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

cm_sk_n=confusion_matrix(LyrFreqDF_2['Decade'],SVM_CM.predict(LyrFreqDF_2.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_CM=svm_test2.fit(LyrFreqDF_2.iloc[:,2:],LyrFreqDF_2['Decade'])
cm_sk=confusion_matrix(LyrFreqDF_2['Decade'],SVM_CM.predict(LyrFreqDF_2.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

cm_sk_n=confusion_matrix(LyrFreqDF_2['Decade'],SVM_CM.predict(LyrFreqDF_2.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_DF=svm_test_model.decision_function(X_test)

SVM_DF[0]

SVM_DF2=svm_test_model2.decision_function(X_test)

SKSVM_topK=top_k_accuracy_score(y_test,SVM_DF,k=3)

SKSVM_topK=top_k_accuracy_score(y_test,SVM_DF2,k=3)

SKSVM_topK

################################################################################
### See if this works on the other frequency distribution ######################
################################################################################

X_train, X_test, y_train, y_test = train_test_split(LyrFreqDF.iloc[:,2:], LyrFreqDF['Decade'])

SKNB_FreqDF=MultinomialNB()
SKNB_FreqDF.fit(X_train,y_train)

print(SKNB_FreqDF.score(X_test,y_test))

cm_sk=confusion_matrix(LyrFreqDF['Decade'],SKNB_FreqDF.predict(LyrFreqDF.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SKNB_FreqDF.classes_)
cm_sk_dsp.plot()
plt.show()

cm_sk_n=confusion_matrix(LyrFreqDF['Decade'],SKNB_FreqDF.predict(LyrFreqDF.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SKNB_FreqDF.classes_)
cm_sk_dsp_n.plot()
plt.show()

SKNB_estimator=SKNB_FreqDF.predict_proba(X_test)
print(SKNB_estimator[0])

SKNB_TopK=top_k_accuracy_score(y_test,SKNB_estimator,k=3)

SKNB_TopK

svm_test=svm.SVC(decision_function_shape='ovr')
svm_test_model=svm_test.fit(X_train,y_train)

print(svm_test_model.score(X_test,y_test))

SVM_CM=svm_test.fit(LyrFreqDF.iloc[:,2:],LyrFreqDF['Decade'])
cm_sk=confusion_matrix(LyrFreqDF['Decade'],SVM_CM.predict(LyrFreqDF.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

cm_sk_n=confusion_matrix(LyrFreqDF['Decade'],SVM_CM.predict(LyrFreqDF.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_DF=svm_test_model.decision_function(X_test)

SVM_DF[0]

SKSVM_TopK=top_k_accuracy_score(y_test,SVM_DF,k=3)

SKSVM_TopK

## cross-validation ##
# this function takes the (number of folds, featureset_dataframe)
# it iterates over the folds, using random sections selected by train_test_split
#   it prints the top k accuracy for each fold (using multinomial naivebayes (MNB) model) and the average top k accuracy at the end
def cross_val_acc_topk_MNB(num_folds, feature_DF):
    subset_size = int(len(feature_DF)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        X_train, X_test, y_train, y_test = train_test_split(feature_DF.iloc[:,2:], feature_DF['Decade'],test_size=subset_size)
        # train using train_this_round
        classifier = SKNB_FreqDF.fit(X_train,y_train)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = top_k_accuracy_score(y_test,classifier.predict_proba(X_test),k=3)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

cross_val_acc_topk_MNB(10,LyrFreqDF)

cross_val_acc_topk_MNB(10,LyrFreqDF_2)

svm_test=svm.SVC(decision_function_shape='ovr')

## cross-validation ##
# this function takes the (number of folds, featureset_dataframe)
# it iterates over the folds, using random sections selected by train_test_split
#   it prints the top k accuracy for each fold using SVM and the average top k accuracy at the end
def cross_val_acc_topk_SVM(num_folds, feature_DF):
    subset_size = int(len(feature_DF)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        X_train, X_test, y_train, y_test = train_test_split(feature_DF.iloc[:,2:], feature_DF['Decade'],test_size=subset_size)
        # train using train_this_round
        classifier = svm_test.fit(X_train,y_train)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = top_k_accuracy_score(y_test,classifier.decision_function(X_test),k=3)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

cross_val_acc_topk_SVM(10,LyrFreqDF)

cross_val_acc_topk_SVM(10,LyrFreqDF_2)

################################################################################
### Begin Feature Set and Modeling Work : POS Tagging ##########################
################################################################################

#LyrPOSDF || POS analysis with no stop words removed

#LyrPOSDF_NLTK || nltk stop words removed

#LyrPOSDF_SPC || spacey stop word removed

#LyrPOSDF_GM || gensim stop word removed

#collect feature set from POS analysis
feature_set_POS=[]
for i in LyrPOSDF.index:
    features_dict={}
    features_item=()
    for h in range(2,len(LyrPOSDF.columns)):
        features_dict[LyrPOSDF.columns[h]]=LyrPOSDF.loc[i,LyrPOSDF.columns[h]]
    features_item=(features_dict,LyrPOSDF.loc[i,'Decade'])
    feature_set_POS.append(features_item)

#now create the test and train set
random.shuffle(feature_set_POS)
cutoff=.80
split=round(len(feature_set_POS)*cutoff)
train_set=feature_set_POS[:split]
test_set=feature_set_POS[split:]

#create the model
POS_train_model=nltk.NaiveBayesClassifier.train(train_set)

#test for accuracy
nltk.classify.accuracy(POS_train_model, test_set)

#collect feature set from POS_NLTK analysis
feature_set_NLTK=[]
for i in LyrPOSDF_NLTK.index:
    features_dict={}
    features_item=()
    for h in range(2,len(LyrPOSDF_NLTK.columns)):
        features_dict[LyrPOSDF_NLTK.columns[h]]=LyrPOSDF_NLTK.loc[i,LyrPOSDF_NLTK.columns[h]]
    features_item=(features_dict,LyrPOSDF_NLTK.loc[i,'Decade'])
    feature_set_NLTK.append(features_item)

#now create the test and train set
random.shuffle(feature_set_NLTK)
cutoff=.80
split=round(len(feature_set_NLTK)*cutoff)
train_set=feature_set_NLTK[:split]
test_set=feature_set_NLTK[split:]

#create the model
NLTK_train_model=nltk.NaiveBayesClassifier.train(train_set)

#test for accuracy
nltk.classify.accuracy(NLTK_train_model, test_set)

#collect feature set from POS_SPC analysis
feature_set_SPC=[]
for i in LyrPOSDF_SPC.index:
    features_dict={}
    features_item=()
    for h in range(2,len(LyrPOSDF_SPC.columns)):
        features_dict[LyrPOSDF_SPC.columns[h]]=LyrPOSDF_SPC.loc[i,LyrPOSDF_SPC.columns[h]]
    features_item=(features_dict,LyrPOSDF_SPC.loc[i,'Decade'])
    feature_set_SPC.append(features_item)

#now create the test and train set
random.shuffle(feature_set_SPC)
cutoff=.80
split=round(len(feature_set_SPC)*cutoff)
train_set=feature_set_SPC[:split]
test_set=feature_set_SPC[split:]

#create the model
SPC_train_model=nltk.NaiveBayesClassifier.train(train_set)

#test for accuracy
nltk.classify.accuracy(SPC_train_model, test_set)

#collect feature set from POS_GM analysis
feature_set_GM=[]
for i in LyrPOSDF_GM.index:
    features_dict={}
    features_item=()
    for h in range(2,len(LyrPOSDF_GM.columns)):
        features_dict[LyrPOSDF_GM.columns[h]]=LyrPOSDF_GM.loc[i,LyrPOSDF_GM.columns[h]]
    features_item=(features_dict,LyrPOSDF_GM.loc[i,'Decade'])
    feature_set_GM.append(features_item)

#now create the test and train set
random.shuffle(feature_set_GM)
cutoff=.80
split=round(len(feature_set_GM)*cutoff)
train_set=feature_set_GM[:split]
test_set=feature_set_GM[split:]

#create the model
GM_train_model=nltk.NaiveBayesClassifier.train(train_set)

#test for accuracy
nltk.classify.accuracy(GM_train_model, test_set)

#perform cross validation
def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

cross_validation_accuracy(10, feature_set_POS)

cross_validation_accuracy(10, feature_set_NLTK)

cross_validation_accuracy(10, feature_set_SPC)

cross_validation_accuracy(10, feature_set_GM)

################################################################################
### Now look at confusion matrix, precision, recall, F1 ########################
################################################################################

#####first looking at POS tag, no stop words#####
#now create the test and train set
random.shuffle(feature_set_POS)
cutoff=.80
split=round(len(feature_set_POS)*cutoff)
train_set=feature_set_POS[:split]
test_set=feature_set_POS[split:]

goldlist = []
predictedlist = []
for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(POS_train_model.classify(features))

# look at the first 30 examples
print(goldlist[:30])
print(predictedlist[:30])

cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))

# or show the results as percentages
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

# call the function with our data
eval_measures(goldlist, predictedlist)

#####looking at POS tag, NLTK stop words#####
#now create the test and train set
random.shuffle(feature_set_NLTK)
cutoff=.80
split=round(len(feature_set_NLTK)*cutoff)
train_set=feature_set_NLTK[:split]
test_set=feature_set_NLTK[split:]

goldlist = []
predictedlist = []
for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(NLTK_train_model.classify(features))

# look at the first 30 examples
print(goldlist[:30])
print(predictedlist[:30])

cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))

# or show the results as percentages
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

# call the function with our data
eval_measures(goldlist, predictedlist)

#####looking at POS tag, spacey stop words#####
#now create the test and train set
random.shuffle(feature_set_SPC)
cutoff=.80
split=round(len(feature_set_SPC)*cutoff)
train_set=feature_set_SPC[:split]
test_set=feature_set_SPC[split:]

goldlist = []
predictedlist = []
for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(SPC_train_model.classify(features))

# look at the first 30 examples
print(goldlist[:30])
print(predictedlist[:30])

cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))

# or show the results as percentages
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

# call the function with our data
eval_measures(goldlist, predictedlist)

#####looking at POS tag, gensim stop words#####
#now create the test and train set
random.shuffle(feature_set_GM)
cutoff=.80
split=round(len(feature_set_GM)*cutoff)
train_set=feature_set_GM[:split]
test_set=feature_set_GM[split:]

goldlist = []
predictedlist = []
for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(GM_train_model.classify(features))

# look at the first 30 examples
print(goldlist[:30])
print(predictedlist[:30])

cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))

# or show the results as percentages
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

# call the function with our data
eval_measures(goldlist, predictedlist)

################################################################################
### Use sklearn and top n predictors instead ###################################
################################################################################

SKMNB=MultinomialNB()

svm_test=svm.SVC(decision_function_shape='ovr')

####first just POS tags
####Multinomial Naivebayes
X_train, X_test, y_train, y_test = train_test_split(LyrPOSDF.iloc[:,2:], LyrPOSDF['Decade'])

SKNB_POS=SKMNB.fit(X_train,y_train)

SKNB_POS_Score=SKNB_POS.score(X_test,y_test)
print('Multinomial Naivebayes Accuracy:', SKNB_POS_Score)

print('\nMultinomial Confusion Matrix:')
cm_sk=confusion_matrix(LyrPOSDF['Decade'],SKNB_POS.predict(LyrPOSDF.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SKNB_POS.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized Multinomial Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrPOSDF['Decade'],SKNB_POS.predict(LyrPOSDF.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SKNB_POS.classes_)
cm_sk_dsp_n.plot()
plt.show()

SKNB_estimatorPOS=SKNB_POS.predict_proba(X_test)

SKNB_POS_topK=top_k_accuracy_score(y_test,SKNB_estimatorPOS,k=3)
print('Multinomial TopK Accuracy:',SKNB_POS_topK)

#####Now SVM
print('\n')
svm_test_model=svm_test.fit(X_train,y_train)
print('SVM Model Accuracy:',svm_test_model.score(X_test,y_test))

print('\nSVM Confusion Matrix:')
SVM_CM=svm_test.fit(LyrPOSDF.iloc[:,2:],LyrPOSDF['Decade'])
cm_sk=confusion_matrix(LyrPOSDF['Decade'],SVM_CM.predict(LyrPOSDF.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized SVM Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrPOSDF['Decade'],SVM_CM.predict(LyrPOSDF.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_DF=svm_test_model.decision_function(X_test)
SKSVM_topK=top_k_accuracy_score(y_test,SVM_DF,k=3)
print('SVM TopK Accuracy:', SKSVM_topK)

####now nltk stop words
####Multinomial Naivebayes
X_train, X_test, y_train, y_test = train_test_split(LyrPOSDF_NLTK.iloc[:,2:], LyrPOSDF_NLTK['Decade'])

SKNB_NLTK=SKMNB.fit(X_train,y_train)

SKNB_NLTK_Score=SKNB_NLTK.score(X_test,y_test)
print('Multinomial Naivebayes Accuracy:', SKNB_NLTK_Score)

print('\nMultinomial Confusion Matrix:')
cm_sk=confusion_matrix(LyrPOSDF_NLTK['Decade'],SKNB_NLTK.predict(LyrPOSDF_NLTK.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SKNB_NLTK.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized Multinomial Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrPOSDF_NLTK['Decade'],SKNB_NLTK.predict(LyrPOSDF_NLTK.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SKNB_NLTK.classes_)
cm_sk_dsp_n.plot()
plt.show()

SKNB_estimatorNLTK=SKNB_NLTK.predict_proba(X_test)

SKNB_NLTK_topK=top_k_accuracy_score(y_test,SKNB_estimatorNLTK,k=3)
print('Multinomial TopK Accuracy:',SKNB_NLTK_topK)

#####Now SVM
print('\n')
svm_test_model=svm_test.fit(X_train,y_train)
print('SVM Model Accuracy:',svm_test_model.score(X_test,y_test))

print('\nSVM Confusion Matrix:')
SVM_CM=svm_test.fit(LyrPOSDF_NLTK.iloc[:,2:],LyrPOSDF_NLTK['Decade'])
cm_sk=confusion_matrix(LyrPOSDF_NLTK['Decade'],SVM_CM.predict(LyrPOSDF_NLTK.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized SVM Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrPOSDF_NLTK['Decade'],SVM_CM.predict(LyrPOSDF_NLTK.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_DF=svm_test_model.decision_function(X_test)
SKSVM_topK=top_k_accuracy_score(y_test,SVM_DF,k=3)
print('SVM TopK Accuracy:', SKSVM_topK)

####now spacey stop words
####Multinomial Naivebayes
X_train, X_test, y_train, y_test = train_test_split(LyrPOSDF_SPC.iloc[:,2:], LyrPOSDF_SPC['Decade'])

SKNB_SPC=SKMNB.fit(X_train,y_train)

SKNB_SPC_Score=SKNB_SPC.score(X_test,y_test)
print('Multinomial Naivebayes Accuracy:', SKNB_SPC_Score)

print('\nMultinomial Confusion Matrix:')
cm_sk=confusion_matrix(LyrPOSDF_SPC['Decade'],SKNB_SPC.predict(LyrPOSDF_SPC.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SKNB_SPC.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized Multinomial Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrPOSDF_SPC['Decade'],SKNB_SPC.predict(LyrPOSDF_SPC.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SKNB_SPC.classes_)
cm_sk_dsp_n.plot()
plt.show()

SKNB_estimatorSPC=SKNB_SPC.predict_proba(X_test)

SKNB_SPC_topK=top_k_accuracy_score(y_test,SKNB_estimatorSPC,k=3)
print('Multinomial TopK Accuracy:',SKNB_SPC_topK)

#####Now SVM
print('\n')
svm_test_model=svm_test.fit(X_train,y_train)
print('SVM Model Accuracy:',svm_test_model.score(X_test,y_test))

print('\nSVM Confusion Matrix:')
SVM_CM=svm_test.fit(LyrPOSDF_SPC.iloc[:,2:],LyrPOSDF_SPC['Decade'])
cm_sk=confusion_matrix(LyrPOSDF_SPC['Decade'],SVM_CM.predict(LyrPOSDF_SPC.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized SVM Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrPOSDF_SPC['Decade'],SVM_CM.predict(LyrPOSDF_SPC.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_DF=svm_test_model.decision_function(X_test)
SKSVM_topK=top_k_accuracy_score(y_test,SVM_DF,k=3)
print('SVM TopK Accuracy:', SKSVM_topK)

####now gensim stop words
####Multinomial Naivebayes
X_train, X_test, y_train, y_test = train_test_split(LyrPOSDF_GM.iloc[:,2:], LyrPOSDF_GM['Decade'])

SKNB_GM=SKMNB.fit(X_train,y_train)

SKNB_GM_Score=SKNB_GM.score(X_test,y_test)
print('Multinomial Naivebayes Accuracy:', SKNB_GM_Score)

print('\nMultinomial Confusion Matrix:')
cm_sk=confusion_matrix(LyrPOSDF_GM['Decade'],SKNB_GM.predict(LyrPOSDF_GM.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SKNB_GM.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized Multinomial Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrPOSDF_GM['Decade'],SKNB_GM.predict(LyrPOSDF_GM.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SKNB_GM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SKNB_estimatorGM=SKNB_GM.predict_proba(X_test)

SKNB_GM_topK=top_k_accuracy_score(y_test,SKNB_estimatorGM,k=3)
print('Multinomial TopK Accuracy:',SKNB_GM_topK)

#####Now SVM
print('\n')
svm_test_model=svm_test.fit(X_train,y_train)
print('SVM Model Accuracy:',svm_test_model.score(X_test,y_test))

print('\nSVM Confusion Matrix:')
SVM_CM=svm_test.fit(LyrPOSDF_GM.iloc[:,2:],LyrPOSDF_GM['Decade'])
cm_sk=confusion_matrix(LyrPOSDF_GM['Decade'],SVM_CM.predict(LyrPOSDF_GM.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized SVM Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrPOSDF_GM['Decade'],SVM_CM.predict(LyrPOSDF_GM.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_DF=svm_test_model.decision_function(X_test)
SKSVM_topK=top_k_accuracy_score(y_test,SVM_DF,k=3)
print('SVM TopK Accuracy:', SKSVM_topK)

##multinomial model cross-validation ##
# this function takes the (number of folds, featureset_dataframe)
# it iterates over the folds, using random sections selected by train_test_split
#   it prints the top k accuracy for each fold (using multinomial naivebayes (MNB) model) and the average top k accuracy at the end
def cross_val_acc_topk_MNB(num_folds, feature_DF):
    subset_size = int(len(feature_DF)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        X_train, X_test, y_train, y_test = train_test_split(feature_DF.iloc[:,2:], feature_DF['Decade'],test_size=subset_size)
        # train using train_this_round
        classifier = SKMNB.fit(X_train,y_train)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = top_k_accuracy_score(y_test,classifier.predict_proba(X_test),k=3)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

#now MNB cross validation
print('POS MultiNB Cross Validation:')
cross_val_acc_topk_MNB(10,LyrPOSDF)

print('\nNLTK MultiNB Cross Validation:')
cross_val_acc_topk_MNB(10,LyrPOSDF_NLTK)

print('\nSPC MultiNB Cross Validation:')
cross_val_acc_topk_MNB(10,LyrPOSDF_SPC)

print('\nGM MultiNB Cross Validation:')
cross_val_acc_topk_MNB(10,LyrPOSDF_GM)

##SVM model cross-validation ##
# this function takes the (number of folds, featureset_dataframe)
# it iterates over the folds, using random sections selected by train_test_split
#   it prints the top k accuracy for each fold using SVM and the average top k accuracy at the end
def cross_val_acc_topk_SVM(num_folds, feature_DF):
    subset_size = int(len(feature_DF)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        X_train, X_test, y_train, y_test = train_test_split(feature_DF.iloc[:,2:], feature_DF['Decade'],test_size=subset_size)
        # train using train_this_round
        classifier = svm_test.fit(X_train,y_train)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = top_k_accuracy_score(y_test,classifier.decision_function(X_test),k=3)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

#now svm cross validation
print('POS SVM Cross Validation:')
cross_val_acc_topk_SVM(10,LyrPOSDF)

print('\nNLTK SVM Cross Validation:')
cross_val_acc_topk_SVM(10,LyrPOSDF_NLTK)

print('\nSPC SVM Cross Validation:')
cross_val_acc_topk_SVM(10,LyrPOSDF_SPC)

print('\nGM SVM Cross Validation:')
cross_val_acc_topk_SVM(10,LyrPOSDF_GM)

################################################################################
### Begin Feature Set and Modeling Work : Sent/Lex Analysis ####################
################################################################################

#LyrSentDF || sent analysis

#LyrEmpDF || analysis using empath

#LyrLiwcDF || analysis usin liwc

#collect feature set from sentiment analysis
feature_set_sent=[]
for i in LyrSentDF.index:
    features_dict={}
    features_item=()
    for h in range(2,len(LyrSentDF.columns)):
        features_dict[LyrSentDF.columns[h]]=LyrSentDF.loc[i,LyrSentDF.columns[h]]
    features_item=(features_dict,LyrSentDF.loc[i,'Decade'])
    feature_set_sent.append(features_item)

feature_set_sent

#now create the test and train set
random.shuffle(feature_set_sent)
cutoff=.80
split=round(len(feature_set_sent)*cutoff)
train_set=feature_set_sent[:split]
test_set=feature_set_sent[split:]

train_set

#create the model
sent_train_model=nltk.NaiveBayesClassifier.train(train_set)

#test for accuracy
nltk.classify.accuracy(sent_train_model, test_set)

#now collect feature set from emp analysis
feature_set_emp=[]
for i in LyrEmpDF.index:
    features_dict={}
    features_item=()
    for h in range(2,len(LyrEmpDF.columns)):
        features_dict[LyrEmpDF.columns[h]]=LyrEmpDF.loc[i,LyrEmpDF.columns[h]]
    features_item=(features_dict,LyrEmpDF.loc[i,'Decade'])
    feature_set_emp.append(features_item)

feature_set_emp

#now create the test and train set
random.shuffle(feature_set_emp)
cutoff=.80
split=round(len(feature_set_emp)*cutoff)
train_set=feature_set_emp[:split]
test_set=feature_set_emp[split:]

#create the model
emp_train_model=nltk.NaiveBayesClassifier.train(train_set)

#test the accuracy
nltk.classify.accuracy(emp_train_model, test_set)

#now collect feature set from liwc analysis
feature_set_liwc=[]
for i in LyrLiwcDF.index:
    features_dict={}
    features_item=()
    for h in range(2,len(LyrLiwcDF.columns)):
        features_dict[LyrLiwcDF.columns[h]]=LyrLiwcDF.loc[i,LyrLiwcDF.columns[h]]
    features_item=(features_dict,LyrLiwcDF.loc[i,'Decade'])
    feature_set_liwc.append(features_item)

feature_set_liwc

#now create the test and train set
random.shuffle(feature_set_liwc)
cutoff=.80
split=round(len(feature_set_liwc)*cutoff)
train_set=feature_set_liwc[:split]
test_set=feature_set_liwc[split:]

#create the model
liwc_train_model=nltk.NaiveBayesClassifier.train(train_set)

#test the accuracy
nltk.classify.accuracy(liwc_train_model, test_set)

#####try combining all three DFs into one, then create a model based on that
SentEmpDF=pd.merge(LyrSentDF,LyrEmpDF,on=['SongID','Decade'])

SentEmpDFLiwc=pd.merge(SentEmpDF,LyrLiwcDF,on=['SongID','Decade'])

SentEmpDFLiwc=SentEmpDFLiwc.fillna(0)

SentEmpDFLiwc

#now collect feature set from all 3 combined
feature_set_3sent=[]
for i in SentEmpDFLiwc.index:
    features_dict={}
    features_item=()
    for h in range(2,len(SentEmpDFLiwc.columns)):
        features_dict[SentEmpDFLiwc.columns[h]]=SentEmpDFLiwc.loc[i,SentEmpDFLiwc.columns[h]]
    features_item=(features_dict,SentEmpDFLiwc.loc[i,'Decade'])
    feature_set_3sent.append(features_item)

#now create the test and train set
random.shuffle(feature_set_3sent)
cutoff=.80
split=round(len(feature_set_3sent)*cutoff)
train_set=feature_set_3sent[:split]
test_set=feature_set_3sent[split:]

#create the model
sent3_train_model=nltk.NaiveBayesClassifier.train(train_set)

#test the accuracy
nltk.classify.accuracy(sent3_train_model, test_set)

SentEmpDFLiwc.to_csv(f'{default_directory}/datafiles/SentEmpDFLiwc.csv',index=False)

## cross-validation ##
# this function takes the number of folds, the feature sets
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the accuracy for each fold and the average accuracy at the end
def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

cross_validation_accuracy(10,feature_set_3sent)

cross_validation_accuracy(10,feature_set_liwc)

cross_validation_accuracy(10,feature_set_emp)

cross_validation_accuracy(10,feature_set_sent)

################################################################################
### Now look at confusion matrix, precision, recall, F1 ########################
################################################################################

#####first looking at sent analysis#####
#now create the test and train set
random.shuffle(feature_set_sent)
cutoff=.80
split=round(len(feature_set_sent)*cutoff)
train_set=feature_set_sent[:split]
test_set=feature_set_sent[split:]

goldlist = []
predictedlist = []
for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(sent_train_model.classify(features))

# look at the first 30 examples
print(goldlist[:30])
print(predictedlist[:30])

cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))

# or show the results as percentages
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

# call the function with our data
eval_measures(goldlist, predictedlist)

#####next looking at emp lexicaon analysis#####
#now create the test and train set
random.shuffle(feature_set_emp)
cutoff=.80
split=round(len(feature_set_emp)*cutoff)
train_set=feature_set_emp[:split]
test_set=feature_set_emp[split:]

goldlist = []
predictedlist = []
for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(emp_train_model.classify(features))

# look at the first 30 examples
print(goldlist[:30])
print(predictedlist[:30])

cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))

# or show the results as percentages
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

# call the function with our data
eval_measures(goldlist, predictedlist)

#####now looking at liwc lexicaon analysis#####
#now create the test and train set
random.shuffle(feature_set_liwc)
cutoff=.80
split=round(len(feature_set_liwc)*cutoff)
train_set=feature_set_liwc[:split]
test_set=feature_set_liwc[split:]

goldlist = []
predictedlist = []
for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(liwc_train_model.classify(features))

# look at the first 30 examples
print(goldlist[:30])
print(predictedlist[:30])

cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))

# or show the results as percentages
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

# call the function with our data
eval_measures(goldlist, predictedlist)

#####now looking at all 3 combined#####
#now create the test and train set
random.shuffle(feature_set_3sent)
cutoff=.80
split=round(len(feature_set_3sent)*cutoff)
train_set=feature_set_3sent[:split]
test_set=feature_set_3sent[split:]

goldlist = []
predictedlist = []
for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(sent3_train_model.classify(features))

# look at the first 30 examples
print(goldlist[:30])
print(predictedlist[:30])

cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))

# or show the results as percentages
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

# call the function with our data
eval_measures(goldlist, predictedlist)

################################################################################
### Use sklearn and top n predictors instead ###################################
################################################################################

SKMNB=MultinomialNB()

svm_test=svm.SVC(decision_function_shape='ovr')

####first sent analysis
####Multinomial Naivebayes

#Multi NB can't take negative numbers, adding 1 to all values in Compound Row
LyrSentDFp1=LyrSentDF.copy(deep=True)
LyrSentDFp1['Compound']=LyrSentDFp1['Compound']+1

X_train, X_test, y_train, y_test = train_test_split(LyrSentDFp1.iloc[:,2:], LyrSentDFp1['Decade'])

SKNB_sent=SKMNB.fit(X_train,y_train)

SKNB_sent_Score=SKNB_sent.score(X_test,y_test)
print('Multinomial Naivebayes Accuracy:', SKNB_sent_Score)

print('\nMultinomial Confusion Matrix:')
cm_sk=confusion_matrix(LyrSentDFp1['Decade'],SKNB_sent.predict(LyrSentDFp1.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SKNB_sent.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized Multinomial Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrSentDFp1['Decade'],SKNB_sent.predict(LyrSentDFp1.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SKNB_sent.classes_)
cm_sk_dsp_n.plot()
plt.show()

SKNB_estimatorSent=SKNB_sent.predict_proba(X_test)

SKNB_Sent_topK=top_k_accuracy_score(y_test,SKNB_estimatorSent,k=3)
print('Multinomial TopK Accuracy:',SKNB_Sent_topK)

#####Now SVM
X_train, X_test, y_train, y_test = train_test_split(LyrSentDF.iloc[:,2:], LyrSentDF['Decade'])
print('\n')
svm_test_model=svm_test.fit(X_train,y_train)
print('SVM Model Accuracy:',svm_test_model.score(X_test,y_test))

print('\nSVM Confusion Matrix:')
SVM_CM=svm_test.fit(LyrSentDF.iloc[:,2:],LyrSentDF['Decade'])
cm_sk=confusion_matrix(LyrSentDF['Decade'],SVM_CM.predict(LyrSentDF.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized SVM Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrSentDF['Decade'],SVM_CM.predict(LyrSentDF.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_DF=svm_test_model.decision_function(X_test)
SKSVM_topK=top_k_accuracy_score(y_test,SVM_DF,k=3)
print('SVM TopK Accuracy:', SKSVM_topK)

####now emp lexicon analysis
####Multinomial Naivebayes
X_train, X_test, y_train, y_test = train_test_split(LyrEmpDF.iloc[:,2:], LyrEmpDF['Decade'])

SKNB_emp=SKMNB.fit(X_train,y_train)

SKNB_emp_Score=SKNB_emp.score(X_test,y_test)
print('Multinomial Naivebayes Accuracy:', SKNB_emp_Score)

print('\nMultinomial Confusion Matrix:')
cm_sk=confusion_matrix(LyrEmpDF['Decade'],SKNB_emp.predict(LyrEmpDF.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SKNB_emp.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized Multinomial Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrEmpDF['Decade'],SKNB_emp.predict(LyrEmpDF.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SKNB_emp.classes_)
cm_sk_dsp_n.plot()
plt.show()

SKNB_estimatorEmp=SKNB_emp.predict_proba(X_test)

SKNB_Emp_topK=top_k_accuracy_score(y_test,SKNB_estimatorEmp,k=3)
print('Multinomial TopK Accuracy:',SKNB_Emp_topK)

#####Now SVM
X_train, X_test, y_train, y_test = train_test_split(LyrEmpDF.iloc[:,2:], LyrEmpDF['Decade'])
print('\n')
svm_test_model=svm_test.fit(X_train,y_train)
print('SVM Model Accuracy:',svm_test_model.score(X_test,y_test))

print('\nSVM Confusion Matrix:')
SVM_CM=svm_test.fit(LyrEmpDF.iloc[:,2:],LyrEmpDF['Decade'])
cm_sk=confusion_matrix(LyrEmpDF['Decade'],SVM_CM.predict(LyrEmpDF.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized SVM Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrEmpDF['Decade'],SVM_CM.predict(LyrEmpDF.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_DF=svm_test_model.decision_function(X_test)
SKSVM_topK=top_k_accuracy_score(y_test,SVM_DF,k=3)
print('SVM TopK Accuracy:', SKSVM_topK)

####now liwc lexicon analysis
####Multinomial Naivebayes
X_train, X_test, y_train, y_test = train_test_split(LyrLiwcDF.iloc[:,2:], LyrLiwcDF['Decade'])

SKNB_liwc=SKMNB.fit(X_train,y_train)

SKNB_liwc_Score=SKNB_liwc.score(X_test,y_test)
print('Multinomial Naivebayes Accuracy:', SKNB_liwc_Score)

print('\nMultinomial Confusion Matrix:')
cm_sk=confusion_matrix(LyrLiwcDF['Decade'],SKNB_liwc.predict(LyrLiwcDF.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SKNB_liwc.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized Multinomial Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrLiwcDF['Decade'],SKNB_liwc.predict(LyrLiwcDF.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SKNB_liwc.classes_)
cm_sk_dsp_n.plot()
plt.show()

SKNB_estimatorLiwc=SKNB_liwc.predict_proba(X_test)

SKNB_Liwc_topK=top_k_accuracy_score(y_test,SKNB_estimatorLiwc,k=3)
print('Multinomial TopK Accuracy:',SKNB_Liwc_topK)

#####Now SVM
X_train, X_test, y_train, y_test = train_test_split(LyrLiwcDF.iloc[:,2:], LyrLiwcDF['Decade'])
print('\n')
svm_test_model=svm_test.fit(X_train,y_train)
print('SVM Model Accuracy:',svm_test_model.score(X_test,y_test))

print('\nSVM Confusion Matrix:')
SVM_CM=svm_test.fit(LyrLiwcDF.iloc[:,2:],LyrLiwcDF['Decade'])
cm_sk=confusion_matrix(LyrLiwcDF['Decade'],SVM_CM.predict(LyrLiwcDF.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized SVM Confusion Matrix:')
cm_sk_n=confusion_matrix(LyrLiwcDF['Decade'],SVM_CM.predict(LyrLiwcDF.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_DF=svm_test_model.decision_function(X_test)
SKSVM_topK=top_k_accuracy_score(y_test,SVM_DF,k=3)
print('SVM TopK Accuracy:', SKSVM_topK)

####now 3sent lexicon analysis
####Multinomial Naivebayes

#adding 1 to compound value from 3 sent model so there are no negatives
SentEmpDFLiwcp1=SentEmpDFLiwc.copy(deep=True)
SentEmpDFLiwcp1['Compound']=SentEmpDFLiwcp1['Compound']+1
X_train, X_test, y_train, y_test = train_test_split(SentEmpDFLiwcp1.iloc[:,2:], SentEmpDFLiwcp1['Decade'])

SKNB_3sent=SKMNB.fit(X_train,y_train)

SKNB_3sent_Score=SKNB_3sent.score(X_test,y_test)
print('Multinomial Naivebayes Accuracy:', SKNB_3sent_Score)

print('\nMultinomial Confusion Matrix:')
cm_sk=confusion_matrix(SentEmpDFLiwcp1['Decade'],SKNB_3sent.predict(SentEmpDFLiwcp1.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SKNB_3sent.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized Multinomial Confusion Matrix:')
cm_sk_n=confusion_matrix(SentEmpDFLiwcp1['Decade'],SKNB_3sent.predict(SentEmpDFLiwcp1.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SKNB_3sent.classes_)
cm_sk_dsp_n.plot()
plt.show()

SKNB_estimator3sent=SKNB_3sent.predict_proba(X_test)

SKNB_3sent_topK=top_k_accuracy_score(y_test,SKNB_estimator3sent,k=3)
print('Multinomial TopK Accuracy:',SKNB_3sent_topK)

#####Now SVM
X_train, X_test, y_train, y_test = train_test_split(SentEmpDFLiwc.iloc[:,2:], SentEmpDFLiwc['Decade'])
print('\n')
svm_test_model=svm_test.fit(X_train,y_train)
print('SVM Model Accuracy:',svm_test_model.score(X_test,y_test))

print('\nSVM Confusion Matrix:')
SVM_CM=svm_test.fit(SentEmpDFLiwc.iloc[:,2:],SentEmpDFLiwc['Decade'])
cm_sk=confusion_matrix(SentEmpDFLiwc['Decade'],SVM_CM.predict(SentEmpDFLiwc.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized SVM Confusion Matrix:')
cm_sk_n=confusion_matrix(SentEmpDFLiwc['Decade'],SVM_CM.predict(SentEmpDFLiwc.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_DF=svm_test_model.decision_function(X_test)
SKSVM_topK=top_k_accuracy_score(y_test,SVM_DF,k=3)
print('SVM TopK Accuracy:', SKSVM_topK)

##multinomial model cross-validation ##
# this function takes the (number of folds, featureset_dataframe)
# it iterates over the folds, using random sections selected by train_test_split
#   it prints the top k accuracy for each fold (using multinomial naivebayes (MNB) model) and the average top k accuracy at the end
def cross_val_acc_topk_MNB(num_folds, feature_DF):
    subset_size = int(len(feature_DF)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        X_train, X_test, y_train, y_test = train_test_split(feature_DF.iloc[:,2:], feature_DF['Decade'],test_size=subset_size)
        # train using train_this_round
        classifier = SKMNB.fit(X_train,y_train)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = top_k_accuracy_score(y_test,classifier.predict_proba(X_test),k=3)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

#now MNB cross validation
print('sent MultiNB Cross Validation:')
cross_val_acc_topk_MNB(10,LyrSentDFp1)

print('\nemp MultiNB Cross Validation:')
cross_val_acc_topk_MNB(10,LyrEmpDF)

print('\nliwc MultiNB Cross Validation:')
cross_val_acc_topk_MNB(10,LyrLiwcDF)

print('\n3sent MultiNB Cross Validation:')
cross_val_acc_topk_MNB(10,SentEmpDFLiwcp1)

##SVM model cross-validation ##
# this function takes the (number of folds, featureset_dataframe)
# it iterates over the folds, using random sections selected by train_test_split
#   it prints the top k accuracy for each fold using SVM and the average top k accuracy at the end
def cross_val_acc_topk_SVM(num_folds, feature_DF):
    subset_size = int(len(feature_DF)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        X_train, X_test, y_train, y_test = train_test_split(feature_DF.iloc[:,2:], feature_DF['Decade'],test_size=subset_size)
        # train using train_this_round
        classifier = svm_test.fit(X_train,y_train)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = top_k_accuracy_score(y_test,classifier.decision_function(X_test),k=3)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

#now SVM cross validation
print('sent SVM Cross Validation:')
cross_val_acc_topk_SVM(10,LyrSentDF)

print('\nemp SVM Cross Validation:')
cross_val_acc_topk_SVM(10,LyrEmpDF)

print('\nliwc SVM Cross Validation:')
cross_val_acc_topk_SVM(10,LyrLiwcDF)

print('\n3sent SVM Cross Validation:')
cross_val_acc_topk_SVM(10,SentEmpDFLiwc)

################################################################################
### Begin Feature Set and Modeling Work : 3 Feat Sets Combined #################
################################################################################

#####try combining all three DFs into one, then create a model based on that
SentPOSDF=pd.merge(SentEmpDFLiwc,LyrPOSDF_GM,on=['SongID','Decade'])

SentPOSFreqDF=pd.merge(SentPOSDF,LyrFreqDF_2,on=['SongID','Decade'])

SentPOSFreqDF

#IMPORTANT NOTE# This code once again took a while to run, so once it finished I
# exported the JSON so that it can just be imported going forward.
# There is code to import it at the top of the page already, the file is feature_set_ALL
'''
#collect feature set from 3 types combined
feature_set_ALL=[]
for i in SentPOSFreqDF.index:
    features_dict={}
    features_item=()
    for h in range(2,len(SentPOSFreqDF.columns)):
        features_dict[SentPOSFreqDF.columns[h]]=SentPOSFreqDF.loc[i,SentPOSFreqDF.columns[h]]
    features_item=(features_dict,SentPOSFreqDF.loc[i,'Decade'])
    feature_set_ALL.append(features_item)
     '''

#####the column 'LinePerSong' was type numpy.int64, which json did not like, so converting them all to regular int
for i in range(0,len(feature_set_ALL)):
    feature_set_ALL[i][0]['LinePerSong']=feature_set_ALL[i][0]['LinePerSong'].item()

'''Exporting it also took a while.
#since it takes so long to create this feature set, export it and then import it
with open(f'{default_directory}/json_files/feature_set_ALL.json', 'w') as file:
    json.dump(feature_set_ALL, file)
    '''

#now create the test and train set
random.shuffle(feature_set_ALL)
cutoff=.80
split=round(len(feature_set_ALL)*cutoff)
train_set=feature_set_ALL[:split]
test_set=feature_set_ALL[split:]

#create the model
ALL_train_model=nltk.NaiveBayesClassifier.train(train_set)

#test for accuracy
nltk.classify.accuracy(ALL_train_model, test_set)

def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

cross_validation_accuracy(10, feature_set_ALL)

################################################################################
### Now look at confusion matrix, precision, recall, F1 ########################
################################################################################

#now create the test and train set
random.shuffle(feature_set_ALL)
cutoff=.80
split=round(len(feature_set_ALL)*cutoff)
train_set=feature_set_ALL[:split]
test_set=feature_set_ALL[split:]

goldlist = []
predictedlist = []
for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(ALL_train_model.classify(features))

# look at the first 30 examples
print(goldlist[:30])
print(predictedlist[:30])

cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))

# or show the results as percentages
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

# call the function with our data
eval_measures(goldlist, predictedlist)

################################################################################
### Use sklearn and top n predictors instead ###################################
################################################################################

SKMNB=MultinomialNB()

svm_test=svm.SVC(decision_function_shape='ovr')

####first just POS tags
####Multinomial Naivebayes
#adding 1 to all compound values since MNB can't handle negative numbers
SentPOSFreqDFp1=SentPOSFreqDF.copy(deep=True)
SentPOSFreqDFp1['Compound']=SentPOSFreqDFp1['Compound']+1
X_train, X_test, y_train, y_test = train_test_split(SentPOSFreqDFp1.iloc[:,2:], SentPOSFreqDFp1['Decade'])

SKNB_ALL=SKMNB.fit(X_train,y_train)

SKNB_ALL_Score=SKNB_ALL.score(X_test,y_test)
print('Multinomial Naivebayes Accuracy:', SKNB_ALL_Score)

print('\nMultinomial Confusion Matrix:')
cm_sk=confusion_matrix(SentPOSFreqDFp1['Decade'],SKNB_ALL.predict(SentPOSFreqDFp1.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SKNB_ALL.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized Multinomial Confusion Matrix:')
cm_sk_n=confusion_matrix(SentPOSFreqDFp1['Decade'],SKNB_ALL.predict(SentPOSFreqDFp1.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SKNB_ALL.classes_)
cm_sk_dsp_n.plot()
plt.show()

SKNB_estimatorALL=SKNB_ALL.predict_proba(X_test)

SKNB_ALL_topK=top_k_accuracy_score(y_test,SKNB_estimatorALL,k=3)
print('Multinomial TopK Accuracy:',SKNB_ALL_topK)

#####Now SVM
X_train, X_test, y_train, y_test = train_test_split(SentPOSFreqDF.iloc[:,2:], SentPOSFreqDF['Decade'])
print('\n')
svm_test_model=svm_test.fit(X_train,y_train)
print('SVM Model Accuracy:',svm_test_model.score(X_test,y_test))

print('\nSVM Confusion Matrix:')
SVM_CM=svm_test.fit(SentPOSFreqDF.iloc[:,2:],SentPOSFreqDF['Decade'])
cm_sk=confusion_matrix(SentPOSFreqDF['Decade'],SVM_CM.predict(SentPOSFreqDF.iloc[:,2:]))
cm_sk_dsp=ConfusionMatrixDisplay(confusion_matrix=cm_sk,display_labels=SVM_CM.classes_)
cm_sk_dsp.plot()
plt.show()

print('\nNormalized SVM Confusion Matrix:')
cm_sk_n=confusion_matrix(SentPOSFreqDF['Decade'],SVM_CM.predict(SentPOSFreqDF.iloc[:,2:]),normalize='all')
cm_sk_dsp_n=ConfusionMatrixDisplay(confusion_matrix=cm_sk_n,display_labels=SVM_CM.classes_)
cm_sk_dsp_n.plot()
plt.show()

SVM_DF=svm_test_model.decision_function(X_test)
SKSVM_topK=top_k_accuracy_score(y_test,SVM_DF,k=3)
print('SVM TopK Accuracy:', SKSVM_topK)

##multinomial model cross-validation ##
# this function takes the (number of folds, featureset_dataframe)
# it iterates over the folds, using random sections selected by train_test_split
#   it prints the top k accuracy for each fold (using multinomial naivebayes (MNB) model) and the average top k accuracy at the end
def cross_val_acc_topk_MNB(num_folds, feature_DF):
    subset_size = int(len(feature_DF)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        X_train, X_test, y_train, y_test = train_test_split(feature_DF.iloc[:,2:], feature_DF['Decade'],test_size=subset_size)
        # train using train_this_round
        classifier = SKMNB.fit(X_train,y_train)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = top_k_accuracy_score(y_test,classifier.predict_proba(X_test),k=3)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

#now MNB cross validation
print('ALL MultiNB Cross Validation:')
cross_val_acc_topk_MNB(10,SentPOSFreqDFp1)

##SVM model cross-validation ##
# this function takes the (number of folds, featureset_dataframe)
# it iterates over the folds, using random sections selected by train_test_split
#   it prints the top k accuracy for each fold using SVM and the average top k accuracy at the end
def cross_val_acc_topk_SVM(num_folds, feature_DF):
    subset_size = int(len(feature_DF)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        X_train, X_test, y_train, y_test = train_test_split(feature_DF.iloc[:,2:], feature_DF['Decade'],test_size=subset_size)
        # train using train_this_round
        classifier = svm_test.fit(X_train,y_train)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = top_k_accuracy_score(y_test,classifier.decision_function(X_test),k=3)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

#now svm cross validation
print('ALL SVM Cross Validation:')
cross_val_acc_topk_SVM(10,SentPOSFreqDF)