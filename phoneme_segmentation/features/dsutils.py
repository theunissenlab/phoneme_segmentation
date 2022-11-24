import numpy as np
import copy
import itertools as itools
import tables
from .DataSequence import DataSequence

DEFAULT_BAD_WORDS = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp","ig","cg", "", "sl", "ns_ap"])

zscore_all = lambda v: (v-v.mean())/v.std()
zscore = lambda v: (v-v.mean(0))/v.std(0)

phonemes = ['AA', 'AE','AH','AO','AW','AY','B','CH','D',
    'DH', 'EH', 'ER',   'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH',
    'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']

def word_length(ds):
    """Number of letters in the DataSequence [ds].
    """
    newdata = np.vstack([len(k) for k in ds.data])
    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def make_word_ds(grids, trfiles, bad_words=DEFAULT_BAD_WORDS):
    """Creates DataSequence objects containing the words from each grid, with any words appearing
    in the [bad_words] set removed.
    """
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[1].make_simple_transcript()
        ## Filter out bad words
        goodtranscript = [x for x in grtranscript
                          if x[2].lower().strip("{}").strip() not in bad_words]
        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        ds[st] = d

    return ds

def make_phoneme_ds(grids, trfiles, bad_words = DEFAULT_BAD_WORDS):
    """Creates DataSequence objects containing the phonemes from each grid.
    """
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[0].make_simple_transcript()
        # filter out bad words
        goodtranscript=np.array([x for x in grtranscript
                        if x[2].lower().strip("{}").strip() not in bad_words])
        for x in goodtranscript:
                x[2] = x[2].upper().strip("0123456789") 
        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        ds[st] = d

    return ds

def make_stress_ds(grids, trfiles, bad_words = DEFAULT_BAD_WORDS):
    """Creates DataSequence objects containing the phonemes from each grid.
    """
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[0].make_simple_transcript()
        # filter out bad words
        goodtranscript=np.array([x for x in grtranscript
                        if x[2].lower().strip("{}").strip() not in bad_words])
        for x in goodtranscript:
            try:
                x[2] = float(x[2][-1])
            except ValueError: 
                x[2] = float(-1)
        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        ds[st] = d

    return ds

def make_phoneWithStress_ds(grids, trfiles, bad_words = DEFAULT_BAD_WORDS):
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[0].make_simple_transcript()
        # filter out bad words
        goodtranscript=np.array([x for x in grtranscript
                        if x[2].lower().strip("{}").strip() not in bad_words])
        for x in goodtranscript:
                x[2] = x[2].upper()
        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        ds[st] = d
        
    return ds

def make_letter_ds(word_seqs, trfiles, wordTime):
    """Creates DataSequence objects containing the biphon combination from each grid.
    """
    ds=dict()
    stories = word_seqs.keys()
    for st in stories:
        word_list = word_seqs[st].data

        word_start_time = word_seqs[st].time_starts
        word_end_time = word_seqs[st].time_ends

        data_new = []
        for w_i, w in enumerate(word_list):
        ## if later need to strip the sign, uncomment this line
        ## currently reserve the sign, because they have linguistic meaning
#             w = np.array([w.replace(s, "") for s in sign_strip])

            this_word_start_time = word_start_time[w_i]
            this_word_end_time = word_end_time[w_i]

            w_len = len(w)
            interval = (this_word_end_time - this_word_start_time)/w_len
            for l_i, l in enumerate(w):
                this_letter_start_time = this_word_start_time+l_i*interval
                this_letter_end_time = this_word_start_time+(l_i+1)*interval
                if wordTime == False:
                    tmp = np.array([this_letter_start_time, this_letter_end_time, l], dtype = object)
                elif wordTime == True:
                    tmp = np.array([this_word_start_time, this_word_end_time, l], dtype = object)
                data_new.append(tmp)
        d = DataSequence.from_grid(np.array(data_new, dtype = object), trfiles[st][0])
        ds[st]  = d
    return ds

def make_diphone_ds(wordseqs, phonseqs, trfiles):
    """Creates DataSequence objects containing the biphon combination from each grid.
    """
    ds = dict()
    stories = wordseqs.keys()
    for st in stories:
        print (st)
        newdata = []
        ## tmp_store save the running index for phon
        tmp_store = 0
        j = 0
        for i,tmax in enumerate(wordseqs[st].time_ends):
            ## index save the index for phon at the end of the word
            #print (tmax) 
            #print (np.where(np.abs(phonseqs[st].time_ends-tmax)<0.2)) 
            index=int(np.where(np.abs(phonseqs[st].time_ends-tmax)<0.1)[0][0])
            #index=int(np.where(np.abs(phonseqs[st].time_ends-tmax)<0.01)[0][0]) 
            
            #index=int(np.where(phonseqs[st].time_ends==tmax)[0])
            if index==tmp_store:
                tmp_biphon_combo = str(phonseqs[st].data[tmp_store]) + ". "
                data = np.array([phonseqs[st].time_starts[tmp_store], phonseqs[st].time_ends[tmp_store], tmp_biphon_combo], dtype = object)
                np.array(newdata.append(data), dtype = object)
            else:
                for j in range(index-tmp_store):
                    if (len(phonseqs[st].data[tmp_store+j])) !=0:
                        tmp_biphon_combo = str(phonseqs[st].data[tmp_store+j]) + "." + str(phonseqs[st].data[tmp_store+j+1])
                        data = np.array([phonseqs[st].time_starts[tmp_store+j], phonseqs[st].time_ends[tmp_store+j+1], tmp_biphon_combo], dtype = object)
                        np.array(newdata.append(data), dtype = object)
            tmp_store=index+1
        d = DataSequence.from_grid(newdata, trfiles[st][0])
        ds[st] = d
    return ds

def make_biphon_wordTime(wordseqs, phonseqs, trfiles):
    """Creates DataSequence objects containing the biphon combination from each grid.
    """
    ds = dict()
    stories = wordseqs.keys()
    for st in stories:
        newdata = []
        ## tmp_store save the running index for phon
        tmp_store = 0
        j = 0
        for i,tmax in enumerate(wordseqs[st].time_ends):
            this_word_startT = wordseqs[st].time_starts[i]
            ## index save the index for phon at the end of the word
            index=int(np.where(phonseqs[st].time_ends==tmax)[0])
            if index==tmp_store:
                        tmp_biphon_combo = str(phonseqs[st].data[tmp_store]) + ". "
                        data = np.array([this_word_startT, tmax, tmp_biphon_combo], dtype = object)
                        np.array(newdata.append(data), dtype = object)
            else:
                for j in range(index-tmp_store):
                    if (len(phonseqs[st].data[tmp_store+j])) !=0:
                        tmp_biphon_combo = str(phonseqs[st].data[tmp_store+j]) + "." + str(phonseqs[st].data[tmp_store+j+1])
                        data = np.array([this_word_startT, tmax, tmp_biphon_combo], dtype = object)
                        np.array(newdata.append(data), dtype = object)
            tmp_store=index+1
        d = DataSequence.from_grid(newdata, trfiles[st][0])
        ds[st] = d
    return ds

def make_biphon_ds_crossWord(word_seqs,  phonseqs, trfiles):
    ds = dict()
    stories = phonseqs.keys()
    for st in stories:
        data = phonseqs[st].data
        phon_start_time = phonseqs[st].time_starts
        phon_end_time = phonseqs[st].time_ends
        
        word_start_time = word_seqs[st].time_starts
        word_end_time = word_seqs[st].time_ends
        
        data_new = []
        ## need to add in space for single word
        for p_i, p in enumerate(data):
            if p_i+1 < len(data):
                ind_word = np.where(word_start_time == phon_start_time[p_i])[0]
                if len(ind_word) == 1: ## if this phon is the beginning of the word
                    this_word_end_time = word_end_time[ind_word]
                    this_phon_end_time = phon_end_time[p_i]
                    if this_word_end_time == this_phon_end_time: # if this word has only one phon
                        combo =  str(p) + ". "
                        tmp = np.array([word_start_time[ind_word], this_word_end_time, combo], dtype = object)
                    else:
                        combo = str(p) + "." + str(data[p_i+1])
                        tmp = np.array([phon_start_time[p_i], phon_start_time[p_i + 1],combo], dtype = object)
                else:
                    combo = str(p) + "." + str(data[p_i+1])
                    tmp = np.array([phon_start_time[p_i], phon_start_time[p_i + 1],combo], dtype = object)
                data_new.append(tmp)
        d = DataSequence.from_grid(np.array(data_new, dtype = object), trfiles[st][0])
        ds[st]  = d
    return ds

def make_triphone_ds(wordseqs, phonseqs, trfiles):
    """Creates DataSequence objects containing the biphon combination from each grid.
    """
    ds = dict()
    stories = wordseqs.keys()
    for st in stories:
        newdata = []
        tmp_store = 0
        for i,tmax in enumerate(wordseqs[st].time_ends):
            #index=int(np.where(phonseqs[st].time_ends==tmax)[0])
            #index=int(np.where(np.abs(phonseqs[st].time_ends-tmax)<0.01)[0][0])
            index=int(np.where(np.abs(phonseqs[st].time_ends-tmax)<0.2)[0][0])
            #if index==tmp_store:
        #               data = np.array([phonseqs[st].time_starts[tmp_store], phonseqs[st].time_ends[tmp_store], phonseqs[st].data[tmp_store]], dtype = object)
        #               np.array(newdata.append(data), dtype = object)
            if index==tmp_store+1:
                        tmp_biphon_combo = str(phonseqs[st].data[tmp_store]) + "." + str(phonseqs[st].data[index]) + ". "
                        data = [phonseqs[st].time_starts[tmp_store], phonseqs[st].time_ends[index], tmp_biphon_combo]
                        np.array(newdata.append(data), dtype = object)
            elif index > tmp_store+1:
                for j in range(index-tmp_store-1):
                    if (len(phonseqs[st].data[tmp_store+j])) !=0:
                        tmp_triphon_combo = str(phonseqs[st].data[tmp_store+j]) + "." + str(phonseqs[st].data[tmp_store+j+1]) + "." + str(phonseqs[st].data[tmp_store+j+2])
                        data = np.array([phonseqs[st].time_starts[tmp_store+j], phonseqs[st].time_ends[tmp_store+j+2], tmp_triphon_combo], dtype=object)
                        np.array(newdata.append(data), dtype = object)
            tmp_store=index+1
        d = DataSequence.from_grid(newdata, trfiles[st][0])
        ds[st] = d
    return ds

def make_NLetter_ds(word_seqs, trfiles, N, wordTime):
    ds=dict()
    stories = word_seqs.keys()
    for st in stories:
        word_list = word_seqs[st].data

        word_start_time = word_seqs[st].time_starts
        word_end_time = word_seqs[st].time_ends

        data_new = []
        for w_i, w in enumerate(word_list):
        ## if later need to strip the sign, uncomment this line
        ## currently reserve the sign, because they have linguistic meaning
#             w = np.array([w.replace(s, "") for s in sign_strip])

            this_word_start_time = word_start_time[w_i]
            this_word_end_time = word_end_time[w_i]

            w_len = len(w)
            
            if w_len <= N:
                tmp = np.array([this_word_start_time, this_word_end_time, w], dtype = object)
                data_new.append(tmp)
                
            elif w_len > N:
                interval = (this_word_end_time - this_word_start_time)/w_len
                for l_i in range(w_len-(N-1)):
                    NLetter = w[l_i:(l_i + N)]
                    this_NLetter_start_time = this_word_start_time+l_i*interval
                    this_NLetter_end_time = this_word_start_time+(l_i+N)*interval
                    if wordTime == False:
                        tmp = np.array([this_NLetter_start_time, this_NLetter_end_time, NLetter], dtype = object)
                    elif wordTime == True:
                        tmp = np.array([this_word_start_time, this_word_end_time, NLetter], dtype = object)
                    data_new.append(tmp)
                    
        d = DataSequence.from_grid(np.array(data_new, dtype = object), trfiles[st][0])
        ds[st]  = d
    return ds

def save_list(data):
    data_tmp = np.hstack([d+"." for d in data])
    s = "".join(data_tmp)
    
    return s

def make_syllables_ds(word_seqs, phonseqs, word_syllable_chunk, trfiles, wordTime): 
    ds=dict()
    stories = word_seqs.keys()
    for st in stories:
        word_list = word_seqs[st].data
        phone_list = phonseqs[st].data
        
        word_start_time = word_seqs[st].time_starts
        word_end_time = word_seqs[st].time_ends
        
        phon_start_time = phonseqs[st].time_starts
        phon_end_time = phonseqs[st].time_ends
        
        data_new = []
        for w_i, w in enumerate(word_list):
            this_word_start_time = word_start_time[w_i]
            this_word_end_time = word_end_time[w_i]
            
            ind_phon_start = int(np.where(phon_start_time == this_word_start_time)[0])
            ind_phon_end = int(np.where(phon_end_time == this_word_end_time)[0])
            
            sIdx = word_syllable_chunk[w+"_"+st+"_"+str(w_i)].astype(int)
            
            if sIdx[0] == 0:
                ## this word has only one syllable
                this_phone_list = phone_list[ind_phon_start:ind_phon_end+1]
                syllable = save_list(this_phone_list)
                tmp = np.array([this_word_start_time, this_word_end_time, syllable], dtype = object)
                data_new.append(tmp)
            else:
                s_count = 0
                for p_i, p in enumerate(range(ind_phon_start,ind_phon_end+1)):
                    if p_i != sIdx[-1] and np.isin(p_i, sIdx) == True:
                    ## when this phoneme is the syllable boundary
                    ## and this phoneme is not the last boundary
                        this_phone_list = phone_list[(ind_phon_start+s_count):(ind_phon_start+p_i)]
                        syllable = save_list(this_phone_list)
                        if wordTime == False:
                            tmp = np.array([phon_start_time[ind_phon_start+s_count], phon_end_time[ind_phon_start+p_i-1], syllable], dtype = object)
                        elif wordTime == True:
                            tmp = np.array([this_word_start_time, this_word_end_time, syllable], dtype = object)
                        data_new.append(tmp) 
                        s_count = p_i
                    elif p_i == sIdx[-1]:
                        ##if this is the last boundary phoneme
                        this_phone_list = phone_list[(ind_phon_start+s_count):(ind_phon_start+p_i)]
                        syllable = save_list(this_phone_list)
                        if wordTime == False:
                            tmp = np.array([phon_start_time[ind_phon_start+s_count], phon_end_time[ind_phon_start+p_i-1], syllable], dtype = object)
                        elif wordTime == True:
                            tmp = np.array([this_word_start_time, this_word_end_time, syllable], dtype = object)
                        data_new.append(tmp)

                        this_phone_list = phone_list[(ind_phon_start+p_i):(ind_phon_end+1)]
                        syllable = save_list(this_phone_list)
                        if wordTime == False:
                            tmp = np.array([phon_start_time[ind_phon_start+p_i], this_word_end_time, syllable], dtype = object)
                        elif wordTime == True:
                            tmp = np.array([this_word_start_time, this_word_end_time, syllable], dtype = object)
                        data_new.append(tmp)
                            
        d = DataSequence.from_grid(np.array(data_new, dtype = object), trfiles[st][0])
        ds[st]  = d
        
    return ds

def make_morpheme_ds(word_seqs, word2morph_table, trfiles, wordTime):
    ds=dict()
    stories = word_seqs.keys()
    for st in stories:
        word_list = word_seqs[st].data

        word_start_time = word_seqs[st].time_starts
        word_end_time = word_seqs[st].time_ends

        data_new = []
        for w_i, w in enumerate(word_list):
            this_word_start_time = word_start_time[w_i]
            this_word_end_time = word_end_time[w_i]

            ## generate morpheme first using polyglot function
            morpheme = word2morph_table[w+"_morpheme"]
            m_len = len(morpheme)

            if m_len == 1:
                ## only one morpheme, the start and end time of morpheme is the same as word
                tmp = np.array([this_word_start_time, this_word_end_time, morpheme[0]], dtype = object)
                data_new.append(tmp)
               
            elif m_len > 1:
                w_len = len(w)
                interval = (this_word_end_time - this_word_start_time)/w_len
                count = 0
                for m_i, m in enumerate(morpheme):
                    m_len = len(m)
                    this_m_start_time = this_word_start_time + interval*(count)
                    this_m_end_time = this_word_start_time + interval*(count+m_len)
                    if wordTime == False:
                        tmp = np.array([this_m_start_time, this_m_end_time, str(m)], dtype = object)
                    elif wordTime == True:
                        tmp = np.array([this_word_start_time, this_word_end_time, str(m)], dtype = object)
                    data_new.append(tmp)
              
                    count = count + m_len

        d = DataSequence.from_grid(np.array(data_new, dtype = object), trfiles[st][0])
        ds[st] = d
    return ds

def return_morpheme_tag(word_seqs, word2morph_table):
    stories = word_seqs.keys()
    tag_all = dict()
    for st in stories:
        word_list = word_seqs[st].data
        tag_tmp = []
        for w_i, w in enumerate(word_list):

            this_parse = word2morph_table[w+"_parse"]
            tag_tmp.extend(["prefix"] * this_parse[0] + ["bound"] * this_parse[1] + ["suffix"] * this_parse[2])
            parse_check = np.sum(this_parse[:3])
            this_m = word2morph_table[w+"_morpheme"]
            if parse_check != len(this_m):
                print (st + "_%s"%(w))
        tag_all[st] = tag_tmp
        
    return tag_all

def categorize_morpheme(morph_seqs, morph_tag):
    '''
    morph_tag is the output of return_morpheme_tag
    '''
    stories = morph_seqs.keys()
    morph_cat = dict()
    for st in stories:
        this_morph_tag = np.array(morph_tag[st])
        all_tags = np.unique(this_morph_tag)
        
        this_morph_seqs = np.array(morph_seqs[st].data)
        seqs_len = this_morph_seqs.shape[0]
        
        for t_i, t in enumerate(all_tags):
            morph_cat[st+"_"+t] = [str("") for x in range(seqs_len)]
            idx = np.where(this_morph_tag == t)[0]
            for i in idx:
                morph_cat[st+"_"+t][i] = this_morph_seqs[i]
            
    return morph_cat

def categorize_morph_ds(morph_cat, morph_seqs, morph_tag, trfiles):
    '''
    morph_cat is the output of categorize_morpheme
    '''
    
    stories = morph_seqs.keys()
    ds = dict()
    for st in stories:
        this_morph_tag = np.array(morph_tag[st])
        all_tags = np.unique(this_morph_tag)
        
        startT = morph_seqs[st].time_starts
        endT = morph_seqs[st].time_ends
        
        for t_i, t in enumerate(all_tags):
            this_morph_cate = morph_cat[st+"_"+t]
            data_new = []

            for s_i, sT in enumerate(startT):
                tmp = np.array([sT, endT[s_i], str(this_morph_cate[s_i])], dtype = object)
                data_new.append(tmp)
            d = DataSequence.from_grid(np.array(data_new, dtype = object), trfiles[st][0])
            ds[st+"_"+t] = d
            
    return ds

def make_character_ds(grids, trfiles):
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[2].make_simple_transcript()
        fixed_grtranscript = [(s,e,map(int, c.split(","))) for s,e,c in grtranscript if c]
        d = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
        ds[st] = d
    return ds

def make_dialogue_ds(grids, trfiles):
    ds = dict()
    for st, gr in grids.iteritems():
        grtranscript = gr.tiers[3].make_simple_transcript()
        fixed_grtranscript = [(s,e,c) for s,e,c in grtranscript if c]
        ds[st] = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
    return ds

def histogram_phonemes(ds, phonemeset=phonemes):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = ds.data
    N = len(ds.data)
    newdata = np.zeros((N, len(phonemeset)))
    phind = dict(enumerate(phonemeset))
    for ii,ph in enumerate(olddata):
        try:
            #ind = phonemeset.index(ph.upper().strip("0123456789"))
            ind = phind[ph.upper().strip("0123456789")]
            newdata[ii][ind] = 1
        except Exception as error:
            pass

    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def histogram_phon(ds, phonemeset):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = np.array([ph.strip("0123456789") for ph in ds.data])
    newdata = np.vstack([olddata==ph for ph in phonemeset]).T
    #newdata = np.vstack([olddata==ph for ph in phonemeset])
    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def make_morph500_model(ds, lsasm):
    newdata = []
    for w in ds.data:
        try:
            w = str.encode(w)
            v = lsasm[w]
        except KeyError as e:
        #except KeyError:
            v = np.zeros((lsasm.data.shape[0],))
        newdata.append(v)
    return DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)



def make_semantic_model(ds, lsasm):
    newdata = []
    for w in ds.data:
        w = w.encode()
        try:
            v = lsasm[w]
        except KeyError as e:
        #except KeyError:
            v = np.zeros((lsasm.data.shape[0],))
        newdata.append(v)
    return DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)

def make_character_model(dss):
    """Make character indicator model for a dict of datasequences.
    """
    stories = dss.keys()
    storychars = dict([(st,np.unique(np.hstack(ds.data))) for st,ds in dss.iteritems()])
    total_chars = sum(map(len, storychars.values()))
    char_inds = dict()
    ncharsdone = 0
    for st in stories:
        char_inds[st] = dict(zip(storychars[st], range(ncharsdone, ncharsdone+len(storychars[st]))))
        ncharsdone += len(storychars[st])

    charmodels = dict()
    for st,ds in dss.iteritems():
        charmat = np.zeros((len(ds.data), total_chars))
        for ti,charlist in enumerate(ds.data):
            for char in charlist:
                charmat[ti, char_inds[st][char]] = 1
        charmodels[st] = DataSequence(charmat, ds.split_inds, ds.data_times, ds.tr_times)

    return charmodels, char_inds

def make_dialogue_model(ds):
    return DataSequence(np.ones((len(ds.data),1)), ds.split_inds, ds.data_times, ds.tr_times)

def modulate(ds, vec):
    """Multiplies each row (each word/phoneme) by the corresponding value in [vec].
    """
    return DataSequence((ds.data.T*vec).T, ds.split_inds, ds.data_times, ds.tr_times)

def catmats(*seqs):
    keys = seqs[0].keys()
    return dict([(k, DataSequence(np.hstack([s[k].data for s in seqs]), seqs[0][k].split_inds)) for k in keys])

def extract_unique_feature(ds):
    stories = ds.keys()
    d = []
    for s in stories:
        d.extend(ds[s].data)
    f = np.unique(np.array(d))
    return f

def letter_count_model(letter_seqs, Rstories, Pstories, model, cci):
    count = dict()
    allstories = Rstories + Pstories
    for story in allstories:
        count[story] = np.array([len(letter) for letter in letter_seqs[story].chunks()])[:, np.newaxis]
    Rstim, Pstim = zscore_ds(count, Rstories, Pstories)

    cci.upload_raw_array("Feature_space/%s_Rstim"%(model), Rstim)
    cci.upload_raw_array("Feature_space/%s_Pstim"%(model), Pstim)
    
def zscore_ds(ds, Rstories, Pstories):
    trim = 5
    Rstim = zscore_all(np.vstack([ds[story][5+trim:-trim] for story in Rstories]))
    Pstim = zscore_all(np.vstack([ds[story][5+trim:-trim] for story in Pstories]))
    return Rstim,  Pstim
    
def generate_FS(ds, func, features, Rstories, Pstories, model, cci):
    interptype = "lanczos" # filter type
    window = 3 # number of lobes in Lanczos filter

    hist = dict()
    downsampled = dict()
    trim = 5

    allstories = Rstories + Pstories
    for story in allstories:
        if model == "stress":
            downsampled[story] = np.nan_to_num(np.array([np.mean(n.astype(float)) for n in ds[story].chunks()])[:, np.newaxis])
        else:
            hist[story] = func(ds[story], features)
            downsampled[story] = hist[story].chunksums(interptype, window=window)

    Rstim = zscore_all(np.vstack([downsampled[story][5+trim:-trim] for story in Rstories]))
    Pstim = zscore_all(np.vstack([downsampled[story][5+trim:-trim] for story in Pstories]))

    print ("%s Rstim "%(model) + str(Rstim.shape))
    print ("%s Pstim "%(model) + str(Pstim.shape))

    return hist, downsampled, Rstim, Pstim    
   # cci.upload_raw_array("Feature_space/%s_Rstim"%(model), Rstim)
   # cci.upload_raw_array("Feature_space/%s_Pstim"%(model), Pstim)


def classify_diphone(word_dic, diphone_dic, option):
	'''
	option 1:
		function: find out the index for single phon word like "I" 
	option 2:
		function: find out the index for short word biphon combo, excluding singlePhon word
	option 3:
		function: find out the index for the biphon combo as the beginning of words
				but the biphon combo themselves are not the words
	option 4: 
        	function: find out the index for the biphon combo as the ending of words
				but the biphon combo themselves are not the words
	word_dic: dictionary containing arrays of start time, end time of each word
	bophon_dict: dictionary containing arrays of start time, end time and data (True/False; presence/absence)
			of each biphon combo
	'''

	phon = []
	phon_ind = []
	
	word = []
	word_ind = []
	
	for t_i, t in enumerate(word_dic.time_starts):
		## find the diphon combo with same start and end time as the words
		ind = int(np.where(diphone_dic.time_starts == t)[0])
		wordT_end = word_dic.time_ends[t_i]
		diphoneT_end = diphone_dic.time_ends[ind]
		
		this_word = word_dic.data[t_i]
		this_diphone = diphone_dic.data[ind]
		
		second_phon = this_diphone.split(".")[-1]
		
		if wordT_end == diphoneT_end:
			if option == 1 and second_phon == " ": ## find out index for single phon word
				phon.append(this_diphone)
				phon_ind.append(ind)
				
				word.append(this_word)
				word_ind.append(t_i)
			elif option == 2 and second_phon != " ": ## find out index for biphon word
				phon.append(this_diphone)
				phon_ind.append(ind)

				word.append(this_word)
				word_ind.append(t_i)
		elif option == 3 and wordT_end > diphoneT_end:
			phon.append(this_diphone)
			phon_ind.append(ind)
			
			word.append(this_word)
			word_ind.append(t_i)
			
		elif option == 4 and wordT_end > diphoneT_end:
		    ind_end = int(np.where(diphone_dic.time_ends == wordT_end)[0])
		    this_diphone = diphone_dic.data[ind_end]
		    phon.append(this_diphone)
		    phon_ind.append(ind_end)

		    word.append(this_word)
		    word_ind.append(t_i)
		
			
	table = np.concatenate((np.array(phon)[:,np.newaxis], np.array(word)[:,np.newaxis], 
							np.array(phon_ind)[:,np.newaxis], np.array(word_ind)[:,np.newaxis]),
									 axis = 1)
	
	return table


def extract_fs(ds, features, table):
	'''
	data [s, f]
		s: each row is a stimulus
		f: each col is a feature
	table: result from classify_biphon 
	function: construct feature space 
		res [f, s]: True false entries
			each row: presence vs absence of a feature
			each col: presence vs absence of a stim
		np.sum(res) should return the number of stim
			could be used as a sanity check
	'''
	
	ind = table[:,2].astype(int)
	phon = table[:,0]
	
	res = copy.deepcopy(ds.data)
	for n_i, n in enumerate(ind):
		i = np.where(ds.data[n] == True)[0]
		if features[i] == phon[n_i]: ## for sanity check purpose
			#print features[i]
			#print table[n_i]
			res[n, i] = False
		else:
			print ("error for %s %s"%(features[ind], table[n_i]))
			
	return DataSequence(res, ds.split_inds, ds.data_times, ds.tr_times)


def phon_word_table(phon_dic, word_dic):
	'''
	For sanity check if the indexing for function classify_biphon is correct or not
	'''

	word = []
	phon = [] 
	word_T = []
	phon_T = []

	phon_ST = phon_dic.time_starts
	word_ST = word_dic.time_starts
	for t_i, t in enumerate(phon_ST):
		ind = np.where(word_ST<=t)[0][-1]
		word.append(word_dic.data[ind])
		phon.append(phon_dic.data[t_i])
		
		phon_T.append(t)
		word_T.append(word_ST[ind])
		
	table = np.concatenate((np.array(phon)[:,np.newaxis], np.array(word)[:,np.newaxis], 
							np.array(phon_T)[:,np.newaxis], np.array(word_T)[:,np.newaxis]),
									 axis = 1)
	
	return table

def matrix_transform(x, mean, std):
	'''
	function: transform count matrix without short words and beginning biphon using biphon fs orig mean and std
	'''
	res = (x-mean)/std
	return res

def save_table_file(filename, filedict):
    """Saves the variables in [filedict] in a hdf5 table file at [filename].
    """
    hf = tables.open_file(filename, mode="w", title="save_file")
    for vname, var in filedict.items():
        hf.create_array("/", vname, var)
    hf.close()


def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)
