# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:23:19 2021

@author: Jorge
"""

import pickle
import matplotlib.pyplot as plt
import pandas as pd
# from langid.langid import LanguageIdentifier, model
import spacy
# from nltk.corpus import stopwords
import re
from gensim import corpora
from gensim.utils import check_output
import os
import shutil
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
import unidecode
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
import datetime
import seaborn as sns
from wordcloud import WordCloud
sns.set()

# Local imports
from topicmodeler.topicmodeling import MalletTrainer

#from nltk.util import ngrams
#from deep_translator import GoogleTranslator

def _preprocessTweet1(tweet):

    # To lower case
    tweet = tweet.lower()

    # Remove URLs
    tweet = re.sub(r'(http|https)://\S+', '', tweet)
    # Remove mails
    tweet = re.sub(r'[A-Za-z1-9-_.]+[@]+[A-Za-z]+.[A-Za-z]+', '', tweet)
    # Remove twitter pics
    tweet = re.sub(r'pic.twitter.com/\S+', '', tweet)
    # Remove usernames and hashtags
    tweet_clean = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.áéíóúñüç]))(@|#)([A-Za-záéíóúñüç]+[A-Za-z0-9-_áéíóúñüç]+)', '', tweet)
    # Remove special characters
    tweet_clean = re.sub(r'[^A-Za-záéíóúñüç]+', ' ', tweet_clean)

    return tweet_clean

def _preprocessTweet2(tweet, nlp, sw, dictErratas):

    # Generate the spacy model
    tweet_nlp = nlp(tweet)

    # Get the lemmas, remove stopwords & normalize chars (diacritics and so)
    tokens = [token for token in tweet_nlp if token.text not in sw]
    tokens = [token.lemma_ for token in tokens if token.tag_ not in ['AUX', 'VERB']]
    tokens = [unidecode.unidecode(token) for token in tokens]
    tokens = [dictErratas.get(token,token) for token in tokens]

    # Regenerate the string
    tweet_final = ' '.join(tokens)

    return tweet_final


class TMtweets():

    def __init__(self, test=False):

        self.df = None
        self.path2project = os.getcwd() + '/LDA'
        self.path2data = self.path2project + '/data'
        self.path2models = self.path2project + '/models'
        self.path2coh = self.path2project + '/coherence'
        self.path2rep = self.path2project + '/report'

        with open(self.path2data + '/stopwords_ext.txt', 'r', encoding='latin1') as f:
            for line in f:
                sw2 = set(line[1:-1].replace('\'', '').split(', '))

        self.stopwords = sw2 #set(stopwords.words('spanish'))
        self.nlp = spacy.load('es_dep_news_trf')
        self.dictionary = None
        self.mallet_path = '/export/usuarios_ml4ds/jarenas/github/TFM_teleco/mallet-2.0.8/bin/mallet'
        # TM hyperparameters
        self.ntopicsArray = list(range(6, 41, 2))
        self.alphasArray = [.5, 1, 5, 10]
        self.nruns = 10
        self.test = test
        self.dictErratas = {'gracia':'gracias', 'mallorco':'mallorca', 'barcelon':'barcelona', \
                            'naturaleza':'natural', 'gastronomico':'gastro', 'gastronomia':'gastro', \
                            'turistico':'turismo', 'historico':'historia', 'artista':'arte', \
                            'artistico':'arte'}

    def loadData(self):

        # New code
        cols2read = ['username', 'created_at', 'text', 'public_metrics.retweet_count', 'public_metrics.reply_count', 'public_metrics.like_count', 'public_metrics.quote_count', 'lang']

        if self.test:
            df = (pd.read_excel(self.path2data + '/tweets_full.xlsx', usecols=cols2read, nrows=100)
                    .dropna(subset=['text'])
                    .reset_index()
                    .rename(columns={'index':'id'}))
        else:
            df = (pd.read_excel(self.path2data + '/tweets_full.xlsx', usecols=cols2read)
                    .dropna(subset=['text'])
                    .reset_index()
                    .rename(columns={'index':'id'}))

        df = df.rename(columns={'username': 'company', 
                                'created_at': 'date', 
                                'text': 'tweet', 
                                'public_metrics.retweet_count': 'n_retweets', 
                                'public_metrics.reply_count': 'n_replies', 
                                'public_metrics.like_count': 'n_likes', 
                                'public_metrics.quote_count': 'n_quoted', 
                                'lang':'language'
                                })

        self.df = df

        print(f'-- Dataset loaded with {len(df)} tweets')


    def preprocessTweets(self):

        print('-- Processing tweets')
        df = self.df.dropna(subset=['tweet'])
        
        # Clean users, hashtags, mails, links...
        df['tweet_lemm'] = df['tweet'].apply(lambda x: _preprocessTweet1(x))
        tweets = df['tweet_lemm'].str.split(' ')
        
        # Get our own bigrams and trigrams (expert knowledge)
        bigram_mod = Phraser(Phrases(''))

        newBigrams = [(b'fin',b'de'), (b'de',b'semana'), (b'medio',b'ambiente'), (b'todo',b'incluido'),     \
                      (b'san',b'valentin'), (b'feliz',b'ano'), (b'viernes',b'santo'), (b'semana',b'santa'), \
                      (b'atencion',b'al'), (b'al',b'cliente'), (b'gran',b'via'), (b'gran',b'canaria')]
            
        for bigram in newBigrams:
            bigram_mod.phrasegrams[bigram] = (50,500)
        
        trigram_mod = Phraser(Phrases(''))
        
        newTrigrams = [(b'fin_de',b'semana'), (b'fin',b'de_semana'), (b'atencion_al',b'cliente'), \
                       (b'atencion',b'al_cliente')]
                      
        for trigram in newTrigrams:
            trigram_mod.phrasegrams[trigram] = (50,500)
            
        tweets = [bigram_mod[tweet] for tweet in tweets]
        tweets = [trigram_mod[tweet] for tweet in tweets]

        df['tweet_lemm'] = [' '.join(tweet) for tweet in tweets]

        # Remove stopwords and lemmatize
        df['tweet_lemm'] = df['tweet_lemm'].apply(lambda x: _preprocessTweet2(x, self.nlp, self.stopwords, self.dictErratas))
        tweets = df['tweet_lemm'].str.split(' ')
        
        # Create the automatic bigram detection
        bigram = Phrases(tweets, min_count=50, threshold=200)  # higher threshold fewer phrases.
        bigram_mod = Phraser(bigram)

        # Apply to the original tweets
        df['tweet_lemm'] = [bigram_mod[tweet] for tweet in tweets]
        df['tweet_lemm'] = df['tweet_lemm'].str.join(' ')
        
        # Filter short tweets
        min_chars = 20
        df = df[df['tweet_lemm'].str.len() >= min_chars]
        
        with open('bigrams.bin','wb') as f:
            pickle.dump(bigram_mod.phrasegrams,f)
            
        with open('trigrams.bin','wb') as f:
            pickle.dump(trigram_mod.phrasegrams,f)
    
        self.df = df


    def translateTweets(self):

        df = self.df

        #identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

        #print('-- Filtering tweets by language')

        #identifier.set_languages(['es', 'en'])
        #df['lang'] = df['tweet'].apply(lambda x: identifier.classify(x))
        #df['lang'], df['pr'] = zip(*df['lang'])

        #df = df[df['pr'] > 0.99]
        #df = df[df['lang'] == 'es']

        # df_es = df[df['lang'] == 'es']
        # df_en = df[df['lang'] == 'en']

        # print('-- Translating tweets in spanish into english')
        # translator = GoogleTranslator(source='es', target='en')
        # df_es['tweet'] = df_es['tweet'].apply(translator.translate)

        # self.df = pd.concat([df_es, df_en], ignore_index=True)

        df = df[df['language'] == 'es']

        self.df = df


    def saveCsvFile(self):

        self.df.to_csv(self.path2data + '/tweets_processed.csv', index=False)


    def saveDictionary(self):

        self.dictionary.save(self.path2data + '/dictionary')


    def loadDictionary(self):

        self.dictionary = Dictionary.load(self.path2data + '/dictionary')


    def filterTokensAndTweets(self):

        df = self.df.dropna(subset=['tweet_lemm'])
        dictionary = corpora.Dictionary()

        # Filter hyperparameters
        no_below = 1
        no_above = 0.99
        keep_n = 500000

        # Generate the vocabulary and filter documents with too few words
        id_lemas = df['tweet_lemm'].str.split().values.tolist()
        # all_lemas = [el for el in id_lemas if len(el) >= min_lemas]
        dictionary.add_documents(id_lemas)

        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        print('-- Filtering lemas that are too common or too rare')
        kept_words = {dictionary[idx] for idx in range(len(dictionary))}
        
        df['tweet_lemm'] = (df['tweet_lemm'].apply(lambda text: text.split())
                                            .apply(lambda text: [el for el in text if el in kept_words])
                                            .apply(lambda text: ' '.join(text)))

        df = df[df['tweet_lemm'].str.split().str.len() >= 3]

        self.df = df
        self.dictionary = dictionary 


    def textProcessPipeline(self):

        self.loadData()
        self.translateTweets()
        self.preprocessTweets()
        self.filterTokensAndTweets()
        self.saveCsvFile()
        self.saveDictionary()


    def generateMalletCorpus(self):

        print('-- Generating txt and mallet corpus')

        # Load the data
        self.df = pd.read_csv(self.path2data + '/tweets_processed.csv', encoding='latin1')
        df = self.df

        # Filter by date
        df = df[df['date'].apply(lambda x: x[:4]) >= '2018']

        # Generating the corpus.txt file and the metadata file
        df[['id', 'tweet_lemm']].to_csv(self.path2data + '/corpus.txt', header=None, index=None, sep=' ', mode='w', encoding='latin1', errors='ignore')
        df['id'].to_csv(self.path2data + '/metadata.csv', index=None)

        # Generating Mallet corpus
        corpus_txt = self.path2data + '/corpus.txt' 
        corpus_mallet = self.path2data + '/corpus.mallet'

        # Command to generate the .mallet
        cmd = self.mallet_path + ' import-file --preserve-case --keep-sequence --input %s --output %s'#--token-regex "[\p{L}\p{M}]+"    
        cmd = cmd % (corpus_txt, corpus_mallet)
        check_output(args=cmd, shell=True)


    def generateTopicModels(self):

        # Other parameters
        num_threads = 4
        num_iterations = 100
        doc_topic_thr = 0
        thetas_thr = 3e-3
        sparse_block = 0
        path2corpus = self.path2data + '/corpus.mallet'
        nmodels = len(self.ntopicsArray) * len(self.alphasArray) * self.nruns

        # To count the number of models created
        counter = 1 
        print('-- Generating topic models')

        #Iterate model training and hyperparameters validation
        for run in range(self.nruns):
            for ntopics in self.ntopicsArray:
                for alpha in self.alphasArray:
                    model_name = '/K_'+str(ntopics)+'_a_'+str(alpha)+'_run_'+str(run)
                    path_model = self.path2models + model_name

                    if os.path.isfile(path_model + '/thetas_dist.pdf'):
                        print(f'   |--> {counter} out of {nmodels} already exists: {model_name}')

                    else:
                        if os.path.isdir(path_model):
                            shutil.rmtree(path_model)

                        os.mkdir(path_model)

                        #Create Topic model
                        MallTr = MalletTrainer(corpusFile=path2corpus, outputFolder=path_model, 
                                               mallet_path=self.mallet_path, numTopics=ntopics, 
                                               alpha=alpha, optimizeInterval=0, 
                                               numThreads=num_threads, numIterations=num_iterations, 
                                               docTopicsThreshold=doc_topic_thr, 
                                               sparse_thr=thetas_thr, sparse_block=sparse_block, 
                                               logger=None)
                        MallTr.fit()

                        print(f'   |--> {counter} out of {nmodels} created: {model_name}')

                    counter += 1


    def getModelsCoherence(self):

        # Load the corpus and the dictionary
        self.loadDictionary()
        dictionary = self.dictionary

        corpus = list()
        with open(self.path2data + '/corpus.txt', 'r') as file:
            for tweet in file:        
                tweet = tweet.replace('\n', '').replace('"', '').split(' ')
                tweet = tweet[2:]
                corpus.append(tweet)

        BoW_corpus = list()

        for i in range(len(corpus)):
            BoW_corpus.append(dictionary.doc2bow(corpus[i], allow_update=True))

        # Generate the model struct to compute results by groups
        TMmodels = os.listdir(self.path2models)

        # Lists to save the results
        n_topics, alpha, n_run, coherence = [], [], [], []
        
        print('-- Computing models coherence')

        for i, TMmodel in enumerate(TMmodels):

            # Take metadata information about the group from the group name
            metadata_i = TMmodel.split('_') 

            n_topics.append(metadata_i[1])
            alpha.append(metadata_i[3])
            n_run.append(metadata_i[5])

            # Get the topic top-words
            topic_keys = list()
            file_topics = f'/K_{metadata_i[1]}_a_{metadata_i[3]}_run_{metadata_i[5]}'

            with open(self.path2models + file_topics + '/topickeys.txt', 'r') as topic_word_file:

                for line in topic_word_file:
                    aux_line = line.replace('\t', ' ').replace('\n', '').split(' ')[2:-1]
                    topic_keys.append(aux_line)

            # Compute the coherence
            print(f'  |--> Computing coherence {i+1} out of {len(TMmodels)}: {file_topics}')
            cm = CoherenceModel(topics=topic_keys, texts=corpus, corpus=BoW_corpus, dictionary=dictionary, coherence='c_v')
            coherence.append(cm.get_coherence())

        # Sort result variables by number of topics
        (n_topics, TMmodels, alpha, n_run, coherence) = tuple(zip(*sorted(zip(n_topics, TMmodels, alpha, n_run, coherence))))

        # Create summary table
        df = pd.DataFrame({'model': TMmodels, 
                           'n_topics': n_topics, 
                           'alpha': alpha, 
                           'n_run': n_run, 
                           'coherence': coherence})

        # Save summary table
        fname = f'/coherence.csv'
        df.to_csv(self.path2coh + fname, index=False)


    def plotCoherence(self):

        # Read the coherence results
        df = pd.read_csv(self.path2coh + '/coherence.csv')

        # Plot
        plt.rc('font', size=15)
        plt.gcf().subplots_adjust(bottom=0.15)

        _, ax = plt.subplots()

        for alpha in self.alphasArray:

            df_f = df[df['alpha'] == alpha]

            x = df_f['n_topics']
            y = df_f['coherence']

            base_line, = ax.plot(x, y, '.', label='')

            df_av = df_f.groupby('n_topics').mean()
            x = df_av.index
            y = df_av['coherence']
            ax.plot(x, y, '.-', label=rf'$\alpha={alpha}$', color=base_line.get_color())

            ax.set_xlabel('No. of topics')
            ax.set_ylabel('Coherence')
            ax.set_title('Models coherence', fontsize=20)
            ax.legend(fontsize=12, loc=0)
            ax.grid()

        # Save figure
        fname = self.path2coh + '/coherence.png'
        plt.savefig(fname, bbox_inches = "tight")
        plt.close()


    def generateReport(self):

        # Selected model hyperparameters
        bestAlpha = 5
        bestK = 14
        bestRun = 2

        # Load the selected model
        modelName = f'K_{bestK}_a_{bestAlpha}_run_{bestRun}'
        
        if os.path.isdir(self.path2rep) == False:
            os.mkdir(self.path2rep)
        
        path_report = self.path2rep + f'/K_bestK_alpha_bestAlpha_run_bestRun'
        
        if os.path.isdir(path_report):
            shutil.rmtree(path_report)

        os.mkdir(path_report)
        os.mkdir(path_report + '/metrics_per_company')
        os.mkdir(path_report + '/topics_distr')
        os.mkdir(path_report + '/topics_evol')
        os.mkdir(path_report + '/topics_words')
        

        # Get the documents weights and word distribution
        with np.load(self.path2models + '/' + modelName + '/modelo.npz') as data:
            # bethas = data['betas_ds']
            thetas = sparse.csr_matrix((data['thetas_data'], data['thetas_indices'], data['thetas_indptr']), shape=data['thetas_shape'])

        # Read the metadata and the processed dataframe
        md = pd.read_csv(self.path2data + '/metadata.csv').values.ravel().tolist()
        df = pd.read_csv(self.path2data + '/tweets_processed.csv', usecols=['id', 'date', 'company'])
        df = df[df['id'].isin(md)]

        # Compute the different plots
        print('-- Generating the report')
        print('  |--> Computing the global topic distribution')
        self.generateTopicsDistr(thetas, 'Topics distribution', 'thetas_dist.png', path_report)
        self.generateTopicsDistrPerCompany(thetas, df, path_report)
        self.generateTopicsEvolution(thetas, df, path_report)
        self.generateWordCloudsPerTopic(modelName, bestK, path_report)

        # Metrics per company and topic
        cols2use = ['company', 'n_retweets', 'n_likes', 'n_replies', 'n_quoted']
        df = pd.read_csv(self.path2data + '/tweets_processed.csv', usecols=cols2use)

        self.generateCompanyMetrics(df)


    def generateTopicsDistr(self, thetas, title, f_name, path):

        # Order topics by weight
        topicNames = range(thetas.shape[1])
        topicWeights = np.sum(thetas, axis=0).tolist()[0]
        (topicWeights, topicNames) = tuple(zip(*sorted(zip(topicWeights, topicNames))))

        # Horizontal bar chart
        # plt.style.use('default')
        plt.barh(topicNames, topicWeights)
        plt.title(title, fontsize=18)
        plt.ylabel('Topic names', fontsize=15)
        plt.xlabel('Topic weights', fontsize=15)
        f_name = path + '/topics_distr/' + f_name
        plt.savefig(f_name, bbox_inches = "tight")
        plt.close()


    def generateTopicsDistrPerCompany(self, thetas, df, path):

        # Filter companies with too few tweets - less than 'threshold'
        threshold = 100

        df_tweets_per_company = df.groupby('company').size().reset_index().rename(columns={0:'n_tweets'})
        df_tweets_per_company = df_tweets_per_company[df_tweets_per_company['n_tweets'] >= threshold]

        # Iterate to get the indices in the DF for each tweet from a company
        compList = df_tweets_per_company['company'].values.tolist()

        for comp in compList:
            print(f'  |--> Computing topic distribution for company {comp}')
            rows = df['company'] == comp
            thetasComp = thetas[rows, :]

            comp2 = comp.replace(' ', '_')
            self.generateTopicsDistr(thetasComp, f'Topics distribution for {comp}', f'thetas_dist_{comp2}.png', path)


    def generateTopicsEvolution(self, thetas, df, path):

        # # Get the year and week of the year to group by
        # df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        # df['week'] = df['date'].dt.isocalendar().week.astype(str)
        # df['year'] = df['date'].dt.isocalendar().year.astype(str)
        # df['date_group'] = df['year'] + '/' + df['week'] + '/0'
        # df['date_group'] = pd.to_datetime(df['date_group'], format='%Y/%W/%w')
        # df = df.drop(columns=['year', 'week'])

        # Get the year and month to group by
        df['date_group'] = df['date'].apply(lambda x: x[:7])
        df['date_group'] = pd.to_datetime(df['date_group'], format='%Y-%m')

        # Get indices by week adns sum the thetas
        dateGroups = (df['date_group'].astype(str)
                                      .apply(lambda x: x[:7])
                                      .unique()
                                      .tolist())

        # Save the results
        thetasEvol = list()
        nTweetsMonth = list()

        for date in dateGroups:

            rows = df['date_group'].astype('str').str.contains(date)
            thetasWeek = np.asarray(np.sum(thetas[rows, :], axis=0)).flatten()
            thetasEvol.append(thetasWeek)
            nTweetsMonth.append(len(rows))

        # Sort by date
        (dateGroups, thetasEvol) = tuple(zip(*sorted(zip(dateGroups, thetasEvol))))

        # Parse dates from string to datetime
        dateGroups = [datetime.datetime.strptime(date, '%Y-%m').date() for date in dateGroups]
    
        # Rearrange the data and normalize
        thetasEvol = np.array(thetasEvol)
        thetasEvol = normalize(thetasEvol, norm='l1', axis=1)

        # Plot the evolution for each topic (not normalized)        
        for i in range(thetas.shape[1]):

            print(f'  |--> Computing topic evolution for topic {i}')

            plt.plot(dateGroups, thetasEvol[:, i])
            plt.title(f'Topic {i}', fontsize=18)
            plt.ylabel('Topic weight', fontsize=15)
            plt.xlabel('Date', fontsize=15)
            plt.xticks(rotation=30)
            f_name = path + f'/topics_evol/topic_{i}_evolution.png'
            plt.savefig(f_name, bbox_inches = "tight")
            plt.close()


    def generateWordCloudsPerTopic(self, modelName, bestK, path):

        # Load the word-count per topic
        df = pd.read_csv(f'{self.path2models}/{modelName}/word-topic-counts.txt', names=['col'])
        df['col'] = df['col'].str.split(' ')
        df = df.set_index(df['col'])
        df['word'] = df['col'].apply(lambda x: x[1])
        df['tuples'] = df['col'].apply(lambda x: x[1:])
        df = df.set_index('word').drop(columns=['col'])

        # Create a dict (word, topic) --> count
        dictList = [dict() for x in range(bestK)]

        tmp = df.values

        for wordTopicCountList in tmp:
            word = wordTopicCountList[0][0]
            topicCountList = wordTopicCountList[0][1:]
            for topicCount in topicCountList:
                topic, count = topicCount.split(':')
                dictList[int(topic)][word] = int(count)

        for i in range(bestK):

            wordcloud = WordCloud(max_font_size=50, max_words=20, background_color="white").fit_words(dictList[i])
            plt.figure()
            plt.title(f'Word cloud for topic {i}', fontsize=18)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            f_name = path + f'/topic_words/topic_{i}_word_cloud.png'
            plt.savefig(f_name, bbox_inches = "tight")
            plt.close()

    # def generateMetricsPerTopic(self, thetas, df):

    #     metrics = df[['n_retweets', 'n_replies', 'n_likes', 'n_quoted']]


    def generateCompanyMetrics(self, df, path):

        print('-- Computing aggregate metrics per company')

        # Filter companies with too few tweets - less than 'threshold'
        threshold = 100

        df_tweets_per_company = df.groupby('company').size().reset_index().rename(columns={0:'n_tweets'})
        df_tweets_per_company = df_tweets_per_company[df_tweets_per_company['n_tweets'] >= threshold]

        df = df.merge(df_tweets_per_company, on='company')

        # Select the metrics to use and aggregate by company
        metrics = ['sum', 'mean', 'max']
        metricsDict = {'n_retweets':metrics, 'n_likes':metrics, 'n_replies':metrics, 'n_quoted':metrics}
        df_agg = df.groupby('company').agg(metricsDict)

        # Save the data
        df_agg.to_csv(f'{path}/metrics_per_company/metrics_per_company.csv')


if __name__ == "__main__":    
    myTM = TMtweets()
    # myTM.textProcessPipeline()
    # myTM.generateMalletCorpus()
    # myTM.generateTopicModels()
    # myTM.getModelsCoherence()
    # myTM.plotCoherence()
    myTM.generateReport()
