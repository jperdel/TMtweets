"""
Created on Sun Nov 30 12:47:13 2014
Modified for Tendencias project on October 2018
Modified for Everis project on March/April 2020
@author: jarenas@ing.uc3m.es
"""

import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
from scipy import sparse
from scipy.spatial.distance import jensenshannon
import pyLDAvis
import matplotlib.pyplot as plt
from gensim.utils import check_output
# from utils.misc import var_num_keyboard, request_confirmation

def lee_vocabfreq(vocabfreq_path):
    """Lee el vocabulario del modelo que se encuentra en el fichero indicado
    Devuelve dos diccionarios, uno usando las palabras como llave, y el otro
    utilizando el id de la palabra como clave
    Devuelve también la lista de frequencias de cada término del vocabulario
    
    Parametro de entrada:
        * vocabfreq_path     : Path con la ruta al vocabulario
    Salida: (vocab_w2id,vocab_id2w)
        * vocab_w2id         : Diccionario {pal_i : id_pal_i}
        * vocab_id2w         : Diccionario {i     : pal_i}
    """
    vocab_w2id = {}
    vocab_id2w = {}
    vocabfreq = []
    with vocabfreq_path.open('r', encoding='utf-8') as f:
        for i,line in enumerate(f):
            wd, freq = line.strip().split('\t')
            vocab_w2id[wd] = i
            vocab_id2w[str(i)] = wd
            vocabfreq.append(int(freq))

    return (vocab_w2id,vocab_id2w,vocabfreq)

def file_len(fname):
    with fname.open('r',encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def largest_indices(ary, n): 
    """Returns the n largest indices from a numpy array.""" 
    flat = ary.flatten() 
    indices = np.argpartition(flat, -n)[-n:] 
    indices = indices[np.argsort(-flat[indices])] 
    idx0, idx1 = np.unravel_index(indices, ary.shape) 
    idx0 = idx0.tolist() 
    idx1 = idx1.tolist() 
    selected_idx = [] 
    for id0, id1 in zip(idx0, idx1): 
        if id0<id1: 
            selected_idx.append((id0, id1, ary[id0,id1])) 
    return selected_idx 


class TMmodel(object):

    #This class represents a Topic Model according to the LDA generative model
    #Essentially the TM is characterized by
    # _alphas: The weight of each topic
    # _betas: The weight of each word in the vocabulary
    # _thetas: The weight of each topic in each document
    #
    # The TM can be trained with Blei's LDA, Mallet, or any other toolbox that
    # produces a model according to this representation
    
    #Estas variables guardarán los valores originales de las alphas, betas, thetas
    #servirán para resetear el modelo después de su optimización
    _betas_orig = None
    _thetas_orig = None
    _alphas_orig = None

    _betas = None
    _thetas = None
    _alphas = None
    _edits = None #Store all editions made to the model
    _ntopics = None
    _betas_ds = None
    _topic_entropy = None
    _descriptions = None
    _vocab_w2id = None
    _vocab_id2w = None
    _vocabfreq = None
    _size_vocab = None
    _vocabfreq_file = None

    def __init__(self, betas=None, thetas=None, alphas=None, 
                     vocabfreq_file=None, from_file=None, logger=None):
        """Inicializacion del model de topicos a partir de las matrices que lo caracterizan
        Ademas de inicializar las correspondientes variables del objeto, se recalcula el Vector
        beta con downscoring (palabras comunes son penalizadas), y se calculan las
        entropias de cada topico.
        :param betas: Matriz numpy de tamaño n_topics x n_words (vocab de cada tópico)
        :param thetas: Matriz numpy de tamaño n_docs x n_topics (composición documental)
        :param alphas: Vector de longitud n_topics, con la importancia de cada perfil
        :param vocabfreq_file: Ruta a un fichero con el vocabulario correspondiente al modelo
                               Contiene también la frecuencia de cada términos del vocabulario
        :param from_file: If not None, contains the name of a file from which the object
                          can be initialized
        """
        if logger:
            self.logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self.logger = logging.getLogger('TMmodel')

        #Convert strings to Paths if necessary
        if vocabfreq_file:
            vocabfreq_file = Path(vocabfreq_file)
        if from_file:
            from_file = Path(from_file)

        #Locate vocabfile for the model
        if not vocabfreq_file and from_file:
            #If not vocabfile was indicated, try to recover from same directory where model is
            vocabfreq_file = from_file.parent.joinpath('vocab_freq.txt')

        if not vocabfreq_file.is_file():
            self.logger.error('-- -- -- It was not possible to locate a valid vocabulary file.')
            self.logger.error('-- -- -- The TMmodel could not be created')
            return

        #Load vocabulary variables from file
        self._vocab_w2id, self._vocab_id2w, self._vocabfreq = lee_vocabfreq(vocabfreq_file)
        self._size_vocab = len(self._vocabfreq)
        self._vocabfreq_file = vocabfreq_file

        #Create model from given data, or recover model from file
        if from_file:
            data = np.load(from_file)
            self._alphas = data['alphas']
            self._betas = data['betas']
            if 'thetas' in data:
                #non-sparse thetas
                self._thetas = data['thetas']
            else:
                self._thetas = sparse.csr_matrix((data['thetas_data'], data['thetas_indices'], data['thetas_indptr']),
                      shape=data['thetas_shape'])
            
            self._alphas_orig = data['alphas_orig']
            self._betas_orig = data['betas_orig']
            if 'thetas_orig' in data:
                self._thetas_orig = data['thetas_orig']
            else:
                self._thetas_orig = sparse.csr_matrix((data['thetas_orig_data'],
                        data['thetas_orig_indices'], data['thetas_orig_indptr']),
                        shape=data['thetas_orig_shape'])
            self._ntopics = data['ntopics']
            self._betas_ds = data['betas_ds']
            self._topic_entropy = data['topic_entropy']
            self._descriptions = [str(x) for x in data['descriptions']]
            self._edits = [str(x) for x in data['edits']]
        else:
            #Cuando el modelo se genera desde el principio, tenemos que
            #guardar los alphas, betas y thetas tanto en las permanentes
            #como en las actuales que se utilizan para visualizar el modelo
            self._betas_orig = betas
            self._thetas_orig = thetas
            self._alphas_orig = alphas
            self._betas = betas
            self._thetas = thetas
            self._alphas = alphas
            self._edits = []
            self._ntopics = self._thetas.shape[1]
            self._calculate_other()
            #Descripciones
            self._descriptions = [x[1] for x in 
                self.get_topic_word_descriptions()]
            #Reordenamiento inicial de tópicos
            self.sort_topics()

        # self.logger.info('-- -- -- Topic model object (TMmodel) successfully created')
        return

    def _calculate_other(self):
        """This function is intended to calculate all other variables
        of the TMmodel object
            * self._betas_ds
            * self._topic_entropy
        """
        #======
        # 1. self._betas_ds
        #Calculamos betas con down-scoring
        self._betas_ds = np.copy(self._betas)
        if np.min(self._betas_ds) < 1e-12:
            self._betas_ds += 1e-12
        deno = np.reshape((sum(np.log(self._betas_ds))/self._ntopics),(self._size_vocab,1))
        deno = np.ones( (self._ntopics,1) ).dot(deno.T)
        self._betas_ds = self._betas_ds * (np.log(self._betas_ds) - deno)
        #======
        # 2. self._topic_entropy
        #Nos aseguramos de que no hay betas menores que 1e-12. En este caso betas nunca es sparse
        if np.min(self._betas) < 1e-12:
            self._betas += 1e-12
        self._topic_entropy = -np.sum(self._betas * np.log(self._betas),axis=1)
        self._topic_entropy = self._topic_entropy/np.log(self._size_vocab)
        return

    def get_alphas(self):
        return self._alphas

    def get_betas(self):
        return self._betas

    def get_thetas(self):
        return self._thetas

    def get_ntopics(self):
        return self._ntopics

    def get_tpc_corrcoef(self):
        #Computes topic correlation. Highly correlated topics
        #co-occure together
        #Topic mean
        med = np.asarray(np.mean(self._thetas,axis=0)).ravel()
        #Topic square mean
        thetas2 = self._thetas.multiply(self._thetas) 
        med2 = np.asarray(np.mean(thetas2,axis=0)).ravel()
        #Topic stds
        stds = np.sqrt(med2-med**2)
        #Topic correlation
        num = self._thetas.T.dot(self._thetas).toarray()/self._thetas.shape[0]
        num = num - med[...,np.newaxis].dot(med[np.newaxis,...])
        deno = stds[...,np.newaxis].dot(stds[np.newaxis,...])
        return num/deno

    def get_tpc_JSdist(self, thr=1e-3):
        #Computes inter-topic distance based on word distributions
        #using Jensen Shannon distance
        # For a more efficient computation with very large vocabularies
        # we implement a threshold for restricting the distance calculation
        # to columns where any elment is greater than threshold thr
        betas_aux = self._betas[:,np.where(self._betas.max(axis=0)>thr)[0]] 
        js_mat = np.zeros((self._ntopics,self._ntopics))
        for k in range(self._ntopics):
            for kk in range(self._ntopics):
                js_mat[k,kk] = jensenshannon(betas_aux[k,:], betas_aux[kk,:])
        return js_mat

    def get_similar_corrcoef(self, npairs):
        #Returns most similar pairs of topics by co-occurence of topics in docs
        corrcoef = self.get_tpc_corrcoef()
        selected = largest_indices(corrcoef, self._ntopics + 2*npairs)
        return selected

    def get_similar_JSdist(self, npairs, thr=1e-3):
        #Returns most similar pairs of topics by co-occurence of topics in docs
        JSsim = 1-self.get_tpc_JSdist(thr)
        selected = largest_indices(JSsim, self._ntopics + 2*npairs)
        return selected

    def get_descriptions(self, tpc=None):
        if not tpc:
            return self._descriptions
        else:
            return self._descriptions[tpc]

    def set_description(self, desc_tpc, tpc):
        """Set description of topic tpc to desc_tpc
        Args:
        :Param desc_tpc: String with the description for the topic
        :Param tpc: Number of topic
        """
        if tpc>self._ntopics-1:
            print('Error setting topic description: Topic ID is larger than number of topics')
        else:
            self._descriptions[tpc] = desc_tpc
        return

    def save_npz(self,npzfile):
        """Salva las matrices que caracterizan el modelo de tópicos en un fichero npz de numpy
        :param npzfile: Nombre del fichero en el que se guardará el modelo
        """
        if isinstance(self._thetas,sparse.csr_matrix):
            np.savez(npzfile,alphas=self._alphas,betas=self._betas,
                 thetas_data=self._thetas.data, thetas_indices=self._thetas.indices,
                 thetas_indptr=self._thetas.indptr, thetas_shape=self._thetas.shape,
                 alphas_orig=self._alphas_orig, betas_orig=self._betas_orig,
                 thetas_orig_data=self._thetas_orig.data, thetas_orig_indices=self._thetas_orig.indices,
                 thetas_orig_indptr=self._thetas_orig.indptr, thetas_orig_shape=self._thetas_orig.shape,
                 ntopics=self._ntopics,betas_ds=self._betas_ds,topic_entropy=self._topic_entropy,
                 descriptions=self._descriptions, edits=self._edits)
        else:
            np.savez(npzfile,alphas=self._alphas,betas=self._betas,thetas=self._thetas,
                 alphas_orig=self._alphas_orig,betas_orig=self._betas_orig,thetas_orig=self._thetas_orig,
                 ntopics=self._ntopics,betas_ds=self._betas_ds,topic_entropy=self._topic_entropy,
                 descriptions=self._descriptions, edits=self._edits)

        if len(self._edits):
            edits_file = Path(npzfile).parent.joinpath('model_edits.txt')
            with edits_file.open('w', encoding='utf8') as fout:
                [fout.write(el+'\n') for el in self._edits]

    def thetas2sparse(self, thr):
        """Convert thetas matrix to CSR format
        :param thr: Threshold to umbralize the matrix
        """
        self._thetas[self._thetas<thr] = 0
        self._thetas = sparse.csr_matrix(self._thetas, copy=True)
        self._thetas = normalize(self._thetas,axis=1,norm='l1')
        self._thetas_orig[self._thetas_orig<thr] = 0
        self._thetas_orig = sparse.csr_matrix(self._thetas_orig, copy=True)
        self._thetas_orig = normalize(self._thetas_orig,axis=1,norm='l1')

    def muestra_perfiles(self,n_palabras=10,tfidf=True,tpc=None):
        """Muestra por pantalla los perfiles del modelo lda por pantalla
        :Param n_palabas: Número de palabras a mostrar para cada perfil
        :Param tfidf: Si True, se hace downscaling de palabras poco
                        específicas (Blei and Lafferty, 2009)
        :Param tpc: If not None, se imprimen los tópicos con ID en la lista tpc
                    e.g.: tpc = [0,3,4]
        """
        if not tpc:
            tpc = range(self._ntopics)
        for i in tpc:
            if tfidf:
                words = [self._vocab_id2w[str(idx2)]
                    for idx2 in np.argsort(self._betas_ds[i])[::-1][0:n_palabras]]
            else:
                words = [self._vocab_id2w[str(idx2)]
                    for idx2 in np.argsort(self._betas[i])[::-1][0:n_palabras]]
            print (str(i)+'\t'+str(self._alphas[i]) + '\t' + ', '.join(words))
        return

    def muestra_descriptions(self,tpc=None,simple=False):
        """Muestra por pantalla las descripciones de los perfiles del modelo lda
        :Param tpc: If not None, se imprimen los tópicos con ID en la lista tpc
                    e.g.: tpc = [0,3,4]
        """
        if not tpc:
            tpc = range(self._ntopics)
        for i in tpc:
            if not simple:
                print (str(i)+'\t'+str(self._alphas[i]) + '\t' + self._descriptions[i])
            else:
                print ('\t'.join(self._descriptions[i].split(', ')))

    def get_topic_word_descriptions(self,n_palabras=15,tfidf=True,tpc=None):
        """Devuelve una lista con las descripciones del modelo de tópicos
        :Param n_palabas: Número de palabras a mostrar para cada perfil
        :Param tfidf: Si True, se hace downscaling de palabras poco
                        específicas (Blei and Lafferty, 2009)
        :Param tpc: If not None, se devuelven las descripciones de los tópicos
                    con ID en la lista tpc e.g.: tpc = [0,3,4]                        
        """
        if not tpc:
            tpc = range(self._ntopics)
        tpc_descs = []
        for i in tpc:
            if tfidf:
                words = [self._vocab_id2w[str(idx2)]
                    for idx2 in np.argsort(self._betas_ds[i])[::-1][0:n_palabras]]
            else:
                words = [self._vocab_id2w[str(idx2)]
                    for idx2 in np.argsort(self._betas[i])[::-1][0:n_palabras]]
            tpc_descs.append((i, ', '.join(words)))
        return tpc_descs

    def most_significant_words_per_topic(self,n_palabras=10,tfidf=True,tpc=None):
        """Devuelve una lista de listas de tuplas, en el formato:
           [  [(palabra1tpc1, beta), (palabra2tpc1, beta)],
              [(palabra1tpc2, beta), (palabra2tpc2, beta)]   ]
        :Param n_palabas: Número de palabras que se devuelven para cada perfil
        :Param tfidf: Si True, para la relevancia se emplea el downscaling
                      de palabras poco específicas de (Blei and Lafferty, 2009)
        :Param tpc: If not None, se devuelven únicamente las descripciones de los
                    tópicos con ID en la lista tpc e.g.: tpc = [0,3,4]                        
        """
        if not tpc:
            tpc = range(self._ntopics)
        mswpt = []
        for i in tpc:
            if tfidf:
                words = [(self._vocab_id2w[str(idx2)], self._betas[i,idx2])
                    for idx2 in np.argsort(self._betas_ds[i])[::-1][0:n_palabras]]
            else:
                words = [(self._vocab_id2w[str(idx2)], self._betas[i,idx2])
                    for idx2 in np.argsort(self._betas[i])[::-1][0:n_palabras]]
            mswpt.append(words)
        return mswpt

    def ndocs_active_topic(self):
        """Returns the number of documents where each topic is active"""
        return (self._thetas != 0).sum(0).tolist()[0]

    def delete_topic(self, tpc):
        """Deletes the indicated topic
        Args:
        :Param tpc: The topic to delete (an integer in range 0:ntopics)
        """
        #Keep record of model changes
        self._edits.append('d ' + str(tpc))
        #Update data matrices
        self._betas = np.delete(self._betas,tpc,0)
        #It could be more efficient, but this complies with full and csr matrices
        tpc_keep = [k for k in range(self._ntopics) if k!=tpc]
        self._thetas = self._thetas[:,tpc_keep]
        self._thetas = normalize(self._thetas,axis=1,norm='l1')
        self._alphas = np.asarray(np.mean(self._thetas,axis=0)).ravel()
        self._ntopics = self._thetas.shape[1]
        
        #Remove topic description
        del self._descriptions[tpc]
        #Recalculate all other stuff
        self._calculate_other()

        return

    def fuse_topics(self, tpcs):
        """Hard fusion of several topics
        Args:
        :Param tpcs: List of topics for the fusion
        """
        #Keep record of model chages
        tpcs = sorted(tpcs)
        self._edits.append('f ' + ' '.join([str(el) for el in tpcs]))
        #Update data matrices. For beta we keep an average of topic vectors
        weights = self._alphas[tpcs]
        bet =  weights[np.newaxis,...].dot(self._betas[tpcs,:])/(sum(weights))
        #keep new topic vector in upper position
        self._betas[tpcs[0],:]=bet
        self._betas = np.delete(self._betas,tpcs[1:],0)
        #For theta we need to keep the sum. Since adding implies changing
        #structure, we need to convert to full matrix first
        #No need to renormalize
        thetas_full = self._thetas.toarray()
        thet = np.sum(thetas_full[:,tpcs],axis=1)
        thetas_full[:,tpcs[0]] = thet
        thetas_full = np.delete(thetas_full,tpcs[1:],1)
        self._thetas = sparse.csr_matrix(thetas_full, copy=True)
        #Compute new alphas and number of topics
        self._alphas = np.asarray(np.mean(self._thetas,axis=0)).ravel()
        self._ntopics = self._thetas.shape[1]

        #Remove topic descriptions
        for tpc in tpcs[1:]:
            del self._descriptions[tpc]
    
        #Recalculate all other stuff
        self._calculate_other()

        return

    def sort_topics(self):
        """Sort topics according to topic size
        """
        # Indexes for topics reordering
        idx = np.argsort(self._alphas)[::-1]
        self._edits.append('s ' + ' '.join([str(el) for el in idx]))

        #Sort data matrices
        self._alphas = self._alphas[idx]
        self._betas = self._betas[idx,:]
        self._thetas = self._thetas[:,idx]

        #Sort topic descriptions
        self._descriptions = [self._descriptions[k] for k in idx.tolist()]
    
        #Recalculate all other stuff
        self._calculate_other()

        return

    def reset_model(self):
        """Resetea el modelo al resultado del LDA original con todos los tópicos"""
        self.__init__(betas=self._betas_orig, thetas=self._thetas_orig,
                      alphas=self._alphas_orig, vocabfreq_file=self._vocabfreq_file)
        return

    def pyLDAvis(self, htmlfile, ndocs, njobs=-1):
        """Generación de la visualización de pyLDAvis
        La visualización se almacena en el fichero que se recibe como argumento
        :Param htmlfile: Path to generated html file
        :Param ndocs: Number of documents used to compute the visualization
        :Param njobs: Number of jobs used to accelerate pyLDAvis
        """
        if len([el for el in self._edits if el.startswith('d')]):
            self.logger.error('-- -- -- pyLDAvis: El modelo ha sido editado y se han eliminado tópicos.')
            self.logger.error('-- -- -- pyLDAvis: No se puede generar la visualización.')
            return

        print('Generating pyLDAvisualization. This is an intensive task, consider sampling number of documents')
        print('The corpus you are using has been trained on', self._thetas.shape[0], 'documents')
        #Ask user for a different number of docs, than default setting in config file
        ndocs = var_num_keyboard('int', ndocs, 
                'How many documents should be used to compute the visualization?')
        if ndocs>self._thetas.shape[0]:
            ndocs = self._thetas.shape[0]
        perm = np.sort(np.random.permutation(self._thetas.shape[0])[:ndocs])
        #We consider all documents are equally important
        doc_len =  ndocs * [1] 
        vocab = [self._vocab_id2w[str(k)] for k in range(len(self._vocab_id2w))] 
        vis_data = pyLDAvis.prepare(self._betas, self._thetas[perm,].toarray(),
                                    doc_len, vocab, self._vocabfreq, lambda_step=0.05,
                                    sort_topics=False, n_jobs=njobs)
        print('Se ha generado la visualización. El fichero se guardará en la carpeta del modelo:')
        print(htmlfile)
        pyLDAvis.save_html(vis_data, htmlfile)
        return 

    def automatic_topic_labeling(self, pathlabeling, ranking='unsupervised', nwords=10,workers=3,
                                    num_candidates=19, num_unsup_labels=5, num_sup_labels=5):
        """Genera vector de descripciones para los tópcios de un modelo
        Las descripciones o títulos de los tópicos se guardan en self._descriptions
        :Param pathlabeling: Root path to NETL files
        :Param ranking: Method to rank candidates ('supervised','unsupervised','both')
        :Param nwords: Number of words for representing a topic.
        :Param workers: Number of workers for parallel computation
        :Param num_candidates: Number of candidates for each topic
        :Param num_unsup_labels: Top N unsupervised labels to propose
        :Param num_sup_labels: Top N supervised labels to propose
        @ Simón Roca Sotelo
        """

        self.logger.info('-- -- -- NETL: Running automatic_topic_labeling ...')
        
        # Make sure pathlabeling is a Path
        pathlabeling = Path(pathlabeling)
        # Relative paths to needed files (pre-trained models)
        doc2vecmodel = pathlabeling.joinpath('pre_trained_models','doc2vec','docvecmodel.d2v')
        word2vecmodel = pathlabeling.joinpath('pre_trained_models','word2vec','word2vec')
        doc2vec_indices_file = pathlabeling.joinpath('support_files','doc2vec_indices')
        word2vec_indices_file = pathlabeling.joinpath('support_files','word2vec_indices')
        # This is precomputed pagerank model needed to genrate pagerank features.
        pagerank_model = pathlabeling.joinpath('support_files','pagerank-titles-sorted.txt')
        # SVM rank classify. After you download SVM Ranker classify gibve the path of svm_rank_classify here
        svm_classify = pathlabeling.joinpath('support_files','svm_rank_classify')
        # This is trained supervised model on the whole our dataset.
        # Run train train_svm_model.py if you want a new model on different dataset. 
        pretrained_svm_model = pathlabeling.joinpath('support_files','svm_model')
        
        # Relative paths to temporal files created during execution.
        out_sup = pathlabeling.joinpath('output_supervised') # The output file for supervised labels.
        data = pathlabeling.joinpath('temp_topics.csv')
        out_unsup = pathlabeling.joinpath('output_unsupervised')
        cand_gen_output = pathlabeling.joinpath('output_candidates')

        # Deleting temporal files if they exist from a previous run.
        temp_files = [cand_gen_output, data, out_sup, out_unsup]
        [f.unlink() for f in temp_files if f.is_file()]

        # Topics to a temporal file.
        descr = [x[1] for x in self.get_topic_word_descriptions(n_palabras=nwords, tfidf=False)]
        with data.open('w', encoding='utf-8') as f:
            head = ['topic_id']
            for n in range(nwords):
                head.append('term'+str(n))
            f.write(','.join(head)+'\n')
            for el in descr:
                f.write(el.replace(', ',','))
                f.write('\n')

        # Calling external script for candidate generation.
        query1 = 'python ' + str(pathlabeling.joinpath('cand_generation.py')) + ' ' + \
                  str(num_candidates) + ' ' + str(doc2vecmodel) + ' ' + str(word2vecmodel) + \
                  ' ' + str(data) + ' ' + str(cand_gen_output) + ' ' + str(doc2vec_indices_file) + \
                  ' ' + str(word2vec_indices_file) + ' ' + str(workers)

        try:
            self.logger.debug('-- -- -- NETL: Extracting candidate labels')
            self.logger.debug('-- -- -- NETL: Query is gonna be: ' + query1)
            check_output(args=query1, shell=True)
        except:
            self.logger.error('-- -- -- NETL failed to extract labels. Revise your command')
            return

        final = []

        # Ranking the previously obtained candidates, in the variants mentioned above.
        try:

            if ranking=='both' or ranking=='supervised':
                query2 = 'python ' + str(pathlabeling.joinpath('supervised_labels.py')) + \
                          ' ' + str(num_sup_labels) + ' ' + str(pagerank_model) + ' ' + \
                          str(data) + ' ' + str(cand_gen_output) + ' ' + str(svm_classify) + \
                          ' ' + str(pretrained_svm_model) + ' ' + str(out_sup)
                try:
                    self.logger.debug('-- -- -- NETL: Executing Supervised Model')
                    self.logger.debug('-- -- -- NETL: Query is gonna be: ' + query2)
                    check_output(args=query2, shell=True)
                except:
                    self.logger.error('-- -- -- NETL failed to extract labels (sup). Revise your command')
                    return

                sup = []
                with out_sup.open('r', encoding='utf-8') as f:
                    for l in f.readlines():
                        sup.append(l.replace('\n','').split(','))
   
            if ranking=='both' or ranking=='unsupervised':
                query3 = 'python ' + str(pathlabeling.joinpath('unsupervised_labels.py')) + \
                          ' ' + str(num_unsup_labels) + ' ' + str(data) + ' ' + \
                          str(cand_gen_output) + ' ' + str(out_unsup)
                try:
                    self.logger.info('-- -- -- NETL Executing Unsupervised model')
                    self.logger.info('-- -- -- NETL: Query is gonna be: ' + query3)
                    check_output(args=query3, shell=True)
                except:
                    self.logger.error('-- -- -- NETL failed to rank labels (unsup). Revise your command')
                    return

                unsup = []
                with out_unsup.open('r', encoding='utf-8') as f:
                    for l in f.readlines():
                        unsup.append(l.replace('\n','').split(','))

            # Joining supervised and unsupervised labels, and getting unique labels.
            for i in range(self._ntopics):
                if ranking=='both':
                    final.append(list(set(sup[i]+unsup[i])))
                elif ranking=='supervised':
                    final.append(list(set(sup[i])))
                elif ranking=='unsupervised':
                    final.append(list(set(unsup[i])))
        except Exception as e:
            self.logger.error('-- -- -- NETL: Something went wrong. Revise the previous log.')

        # Deleting temporal files at the end
        self.logger.debug('-- -- -- NETL: Deleting temporal files')
        [f.unlink() for f in temp_files if f.is_file()]

        if len(final)>0:
            for k, wds in enumerate(final):
                proposed = ', '.join(wds)
                print(10*'=')
                print('Topic ', k)
                print('Current description is', self._descriptions[k])
                print('Proposed description is', wds)
                print('\n')
                if request_confirmation(msg='Keep newly proposed description?'):
                    self._descriptions[k] = proposed
        return


class MalletTrainer(object):
    
    def __init__(self, corpusFile, outputFolder, mallet_path, 
        numTopics=None, alpha=None, optimizeInterval=None,
        numThreads=None, numIterations=None, docTopicsThreshold=None,
        sparse_thr=None, sparse_block=0, logger=None):
        """Inicializador del objeto
        """
        self._corpusFile = Path(corpusFile)
        self._numTopics = numTopics
        self._alpha = alpha
        self._optimizeInterval = optimizeInterval
        self._numThreads = numThreads
        self._numIterations = numIterations
        self._docTopicsThreshold = docTopicsThreshold
        self._outputFolder = Path(outputFolder)
        self._sparse_thr = sparse_thr
        self._sparse_block = sparse_block
        self._mallet_path = Path(mallet_path)
        if logger:
            self.logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self.logger = logging.getLogger('MalletTrainer')

    def adj_settings(self):
        """Ajuste de parámetros manual"""
        self._numTopics = var_num_keyboard('int', self._numTopics,
                                    'Número de tópicos para el modelo')
        self._alpha = var_num_keyboard('float',1,
                                    'Valor para el parametro alpha')
        self._optimizeInterval = var_num_keyboard('int',self._optimizeInterval,
                                'Optimization of hyperparameters every optimize_interval iterations')
        self._numIterations = var_num_keyboard('int',self._numIterations,
                                'Iteraciones máximas para el muestreo de Gibbs')
        self._sparse_thr = var_num_keyboard('float',self._sparse_thr,
                                'Probabilidad para poda para "sparsification" del modelo')

    def fit(self):
        """Rutina de entrenamiento
        """
        config_file = self._outputFolder.joinpath('train.config')
        with config_file.open('w', encoding='utf8') as fout:
            fout.write('input = ' + self._corpusFile.as_posix() + '\n')
            fout.write('num-topics = ' + str(self._numTopics) + '\n')
            fout.write('alpha = ' + str(self._alpha) + '\n')
            fout.write('optimize-interval = ' + str(self._optimizeInterval) + '\n')
            fout.write('num-threads = ' + str(self._numThreads) + '\n')
            fout.write('num-iterations = ' + str(self._numIterations) + '\n')
            fout.write('doc-topics-threshold = ' + str(self._docTopicsThreshold) + '\n')
            #fout.write('output-state = ' + os.path.join(self._outputFolder, 'topic-state.gz') + '\n')
            fout.write('output-doc-topics = ' + \
                self._outputFolder.joinpath('doc-topics.txt').as_posix() + '\n')
            fout.write('word-topic-counts-file = ' + \
                self._outputFolder.joinpath('word-topic-counts.txt').as_posix() + '\n')
            fout.write('diagnostics-file = ' + \
                self._outputFolder.joinpath('diagnostics.xml ').as_posix() + '\n')
            fout.write('xml-topic-report = ' + \
                self._outputFolder.joinpath('topic-report.xml').as_posix() + '\n')
            fout.write('output-topic-keys = ' + \
                self._outputFolder.joinpath('topickeys.txt').as_posix() + '\n')
            fout.write('inferencer-filename = ' + \
                self._outputFolder.joinpath('inferencer.mallet').as_posix() + '\n')
            #fout.write('output-model = ' + \
            #    self._outputFolder.joinpath('modelo.bin').as_posix() + '\n')
            #fout.write('topic-word-weights-file = ' + \
            #    self._outputFolder.joinpath('topic-word-weights.txt').as_posix() + '\n')

        cmd = str(self._mallet_path) + ' train-topics --config ' + str(config_file)

        try:
            # self.logger.info(f'-- -- Training mallet topic model. Command is {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Model training failed. Revise command')
            return

        thetas_file = self._outputFolder.joinpath('doc-topics.txt')
        #Modified to allow for non integer identifier
        cols = [k for k in np.arange(2,self._numTopics+2)]

        if self._sparse_block==0:
            self.logger.debug('-- -- Sparsifying doc-topics matrix')
            thetas32 = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32, usecols=cols)
            #thetas32 = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32)[:,2:]
            #Save figure to check thresholding is correct
            allvalues = np.sort(thetas32.flatten())
            step = int(np.round(len(allvalues)/1000))
            plt.semilogx(allvalues[::step], (100/len(allvalues))*np.arange(0,len(allvalues))[::step])
            plt.semilogx([self._sparse_thr, self._sparse_thr], [0,100], 'r')
            plot_file = self._outputFolder.joinpath('thetas_dist.pdf')
            plt.savefig(plot_file)
            plt.close()
            #sparsify thetas
            thetas32[thetas32<self._sparse_thr] = 0
            thetas32 = normalize(thetas32,axis=1,norm='l1')
            thetas32_sparse = sparse.csr_matrix(thetas32, copy=True)

        else:
            self.logger.debug('-- -- Sparsifying doc-topics matrix using blocks')
            #Leemos la matriz en bloques
            ndocs = file_len(thetas_file)
            thetas32 = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32,
                                     usecols=cols, max_rows=self._sparse_block)
            #Save figure to check thresholding is correct
            #In this case, the figure will be calculated over just one block of thetas
            allvalues = np.sort(thetas32.flatten())
            step = int(np.round(len(allvalues)/1000))
            plt.semilogx(allvalues[::step], (100/len(allvalues))*np.arange(0,len(allvalues))[::step])
            plt.semilogx([self._sparse_thr, self._sparse_thr], [0,100], 'r')
            plot_file = self._outputFolder.joinpath('thetas_dist.pdf')
            plt.savefig(plot_file)
            plt.close()
            #sparsify thetas
            thetas32[thetas32<self._sparse_thr] = 0
            thetas32 = normalize(thetas32,axis=1,norm='l1')
            thetas32_sparse = sparse.csr_matrix(thetas32, copy=True)
            for init_pos in np.arange(0,ndocs,self._sparse_block)[1:]:
                thetas32_b = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32,
                                         usecols=cols, max_rows=self._sparse_block,
                                         skiprows=init_pos)
                #sparsify thetas
                thetas32_b[thetas32_b<self._sparse_thr] = 0
                thetas32_b = normalize(thetas32_b,axis=1,norm='l1')
                thetas32_b_sparse = sparse.csr_matrix(thetas32_b, copy=True)
                thetas32_sparse = sparse.vstack([thetas32_sparse, thetas32_b_sparse])

        #Recalculamos alphas para evitar errores de redondeo por la sparsification
        alphas = np.asarray(np.mean(thetas32_sparse,axis=0)).ravel()

        #Create vocabulary files
        wtcFile = self._outputFolder.joinpath('word-topic-counts.txt')
        vocab_size = file_len(wtcFile)
        betas = np.zeros((self._numTopics,vocab_size))
        vocab = []
        term_freq = np.zeros((vocab_size,))

        with wtcFile.open('r', encoding='utf8') as fin:
            for i,line in enumerate(fin):
                elements = line.split()
                vocab.append(elements[1])
                for counts in elements[2:]:
                    tpc = int(counts.split(':')[0])
                    cnt = int(counts.split(':')[1])
                    betas[tpc,i] += cnt
                    term_freq[i] += cnt
        betas = normalize(betas,axis=1,norm='l1')
        #save vocabulary and frequencies
        with self._outputFolder.joinpath('vocab.txt').open('w', encoding='utf8') as fout:
            [fout.write(el+'\n') for el in vocab]
        with self._outputFolder.joinpath('vocab_freq.txt').open('w', encoding='utf8') as fout:
            [fout.write(el[0]+'\t'+str(int(el[1]))+'\n') for el in zip(vocab,term_freq)]
        self.logger.debug('-- -- Mallet training: Vocabulary files generated')

        tmodel = TMmodel(betas=betas,thetas=thetas32_sparse,alphas=alphas,
                            vocabfreq_file=self._outputFolder.joinpath('vocab_freq.txt'),
                            logger=self.logger)
        tmodel.save_npz(self._outputFolder.joinpath('modelo.npz'))

        #Remove doc-topics file. It is no longer needed
        thetas_file.unlink()

        return

