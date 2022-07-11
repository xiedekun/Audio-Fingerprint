# Author: Dekun Xie #

import os 
import IPython.display as ipd
import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
import pickle
import hashlib

targets = []
queries = []
answers = []
fanout=25 
anchor_distance=8

def fingerprintBuilder(data_path, fingerprint_path):
    '''Generate fingerprint of the target databases.
        If the fingerprint exists, it will load that directly instead of generating'''
    global targets
    global fanout
    global anchor_distance
    targetDir = data_path
    fingerprintDir = fingerprint_path      

    targets  = get_files(targetDir)
    coordinates_targets = []

    for entry in os.scandir(fingerprintDir):
        if entry.name=='target_address.data':
            print(f'Fingerprints(target_address.data) already exists in folder \'{fingerprint_path}\'')
            return
        
    for i, target in enumerate(targets):
        stft = calculate_stft(target)
        coordinates_targets.append(calculate_contellation_map(stft,min_distance=8))
        print(f'Generating Contellation Map {i+1}/{len(targets)}...')
    
    target_address = generate_target_hash(coordinates_targets,fanout=fanout, anchor_distance=anchor_distance)
    with open(os.path.join(fingerprintDir,'target_address.data'), 'wb') as f:
        print(f'saving fingerprints in {fingerprintDir}/target_address.data...')
        pickle.dump(target_address, f)
        print(f'fingerprints saved')

def audioIdentification(query_path, fingerprint_path, output_path):
    '''Generate fingerprint of the query data to match.
    If the fingerprint exists, it will load that directly instead of generating.
    Also, the idetification results will be produced in .txt file'''
    global targets
    global answers
    global queries
    global fanout
    global anchor_distance

    queryDir = query_path
    queries = get_files(queryDir)
    answers = get_answers(targets, queries)

    target_address = []
    coordinates_query = []

    with open(os.path.join(fingerprint_path,'target_address.data'), 'rb') as f:
        print(f'Loading fingerprints(target_address.data) in folder \'{fingerprint_path}\'...')
        target_address = pickle.load(f)
        

    for i, q in enumerate(queries):
        stft = calculate_stft(q)
        coordinates_query.append(calculate_contellation_map(stft,min_distance=8))
        print(f'Generating query Contellation Map {i+1}/{len(queries)}...')

    finger_match(coordinates_query, target_address, fanout=fanout, anchor_distance=anchor_distance, output_path=output_path)
    print('Finished')
    
def get_files(dir_path):
    files = []
    for entry in os.scandir(dir_path):
        if entry.name.endswith('.wav'):
            path = os.path.join(dir_path,entry.name)
            files.append(path)
    return files
        
def get_file_name(file):
    name = file.split('\\')
    return name[-1][0:-4]

def get_answers(targets, queries):
    answers = dict()
    for query in queries:
        query_name = get_file_name(query)
        for target in targets:
            target_name = get_file_name(target)
            if target_name in query_name:
                answers.update({query_name: target_name})    
    return answers


def calculate_stft(file, plot=False):
    y, sr = librosa.load(file)
    # compute and plot STFT spectrogram
    D = np.abs(librosa.stft(normalise(y), n_fft=1024, window='hann', win_length=1024, hop_length=512))
    # D = librosa.feature.melspectrogram(y,sr,n_fft=2048, window='hann', win_length=1024,hop_length=512)
    if plot:
        plt.figure(figsize=(10,5))
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='linear',
                                x_axis='time', cmap='gray_r', sr=sr)
    return D

def calculate_contellation_map(D, min_distance=10, plot=False):
    # detect peaks from STFT and plot contellation map
    coordinates = peak_local_max(np.log(D), min_distance=min_distance, threshold_rel=0.05, indices=False)
    if plot:
        plt.figure(figsize=(10,5))
        plt.imshow(coordinates, cmap=plt.cm.gray_r, origin='lower')
    
    return coordinates

def normalise(wave):
    wave = (wave - np.min(wave))/(np.max(wave) - np.min(wave))
    return wave

def generate_target_hash(coordinates_targets, fanout=30, anchor_distance=10):
    target_address = dict()
    for song_id, coordinates in enumerate(coordinates_targets):
        print(f'Generating Hash {song_id+1}/{len(targets)}...')
        contellation = np.where(coordinates.T==True)
        for i in range(len(contellation[0]) - fanout - anchor_distance + 1):
            anchor_time = contellation[0][i]
            anchor_frequency = contellation[1][i]
            for j in range(fanout):
                time = contellation[0][i+j+anchor_distance]
                frequency = contellation[1][i+j+anchor_distance]
                hash_ = hashlib.sha256(str((anchor_frequency, frequency, time - anchor_time)).encode()).hexdigest()
                if target_address.__contains__(hash_):
                    value = target_address[hash_]
                    value.append((anchor_time, song_id))
                    target_address.update({hash_:value})
                else:
                    target_address.update({hash_:[(anchor_time, song_id)]})
    return target_address

def generate_query_hash(coordinates_query, fanout=30, anchor_distance=10):
    for query_id, coordinates in enumerate(coordinates_query):
        print(f'Generating Query Hash {query_id+1}/{len(queries)}...')
        query_address = []
        contellation = np.where(coordinates.T==True)
        for i in range(len(contellation[0]) - fanout - anchor_distance + 1):
            anchor_time = contellation[0][i]
            anchor_frequency = contellation[1][i]
            for j in range(fanout):
                time = contellation[0][i+j+anchor_distance]
                frequency = contellation[1][i+j+anchor_distance]
                hash_ = hashlib.sha256(str((anchor_frequency, frequency, time - anchor_time)).encode()).hexdigest()
                query_address.append((anchor_time, hash_))
        yield query_address

def finger_match(coordinates_query, target_address, fanout=30, anchor_distance=10, top=3, evaluation=False, output_path =None):
    maps_query = []
    max_f_query = []
    if output_path:
        with open(output_path,'w+') as f:
            content = ''
            f.writelines(content)
            
    for query_id, query_list in enumerate(generate_query_hash(coordinates_query)):
        print(f'Query: {get_file_name(queries[query_id])}')
        cur_queries = answers[get_file_name(queries[query_id])]
                
        if len(query_list) == 0:
            if evaluation:
                maps_query.append(0)
                max_f_query.append(0)
            if output_path:
                with open(output_path,'a') as f:
                    content = f'{cur_queries}.wav\tno matches\n'
                    f.writelines(content)
            print('Sorry, the query finger is empty\n')
            continue

        indicator = dict() 

        for item in query_list:
            n, h = item
            if target_address.__contains__(h):
                l = np.array(target_address[h])
                time = np.dstack(l)[0][0] - n
                song_id = np.dstack(l)[0][1]
                keys = np.dstack((time, song_id))[0]
                for key in keys:
                    key = tuple(key)
                    if indicator.__contains__(key):
                        indicator.update({key: indicator[key]+1})
                    else:
                        indicator.update({key: 1})
        # results = sorted(indicator.items(), key=lambda x:x[1], reverse=True)
        # print(f'Results: {indicator.items()}\n')
        # filter(lambda x: x>=4, indicator)
        result_table = dict()
        for item in indicator.items():
            song_id = item[0][1]
            count_number = item[1]
            if result_table.__contains__(song_id):             
                result_table.update({song_id: result_table[song_id]+count_number})
            else:
                result_table.update({song_id: count_number})
        results = sorted(result_table.items(), key=lambda x:x[1], reverse=True)
        res = []
        for result in results[:top]:
            res.append(get_file_name(targets[result[0]]))
        print(res,'\n', results[:top],'\n')  
        
        if output_path:
            res1 = get_file_name(targets[results[0][0]])
            res2 = get_file_name(targets[results[1][0]])
            res3 = get_file_name(targets[results[2][0]])
            with open(output_path,'a') as f:
                content = f'{cur_queries}.wav\t{res1}.wav\t{res3}.wav\t{res3}.wav\n'
                f.writelines(content)
        if evaluation:
            relevance = relevance_function(results, cur_queries)
            map_ = MAP(results[:top], relevance)
            maps_query.append(map_)    

            fmax = max_f_measure(results[:top], relevance)
            max_f_query.append(fmax)

    return maps_query, max_f_query

# evaluation
def relevance_function(rank_data, queries):
    relevance = []
    for data in rank_data:
        if get_file_name(targets[data[0][1]]) == queries:
            relevance.append(1)
        else:
            relevance.append(0)
    return np.array(relevance)


def get_precision(rank, relevance):
    precision = []
    for r in range(len(rank)):
        sigma = 0.0
        for r in range(r+1):
            sigma += relevance[r]
        precision.append(sigma/(r+1))
    return np.array(precision)

def get_recall(rank, relevance):
    recall = []
    
    for r in range(len(rank)):
        sigma = 0.0
        for r in range(r+1):
            sigma += relevance[r]
        recall.append(sigma/len(relevance[relevance==1]))
    return np.nan_to_num(np.array(recall))

def f_measure(rank, relevance):
    precision = get_precision(rank, relevance)
    recall = get_recall(rank, relevance)
    f = 2 * precision * recall / (precision + recall)
    return np.nan_to_num(f)

def MAP(rank, relevance):
    precision = get_precision(rank, relevance)
    p_hat_q = []
    for r in range(len(rank)):
        sigma = 0.0
        for r in range(r+1):
            sigma += relevance[r] * precision[r]
        p_hat_q.append(sigma/len(relevance[relevance==1]))
        
    p_hat = np.array(p_hat_q).mean()
    
    return np.nan_to_num(p_hat)

def max_f_measure(rank, relevance):
    f = f_measure(rank, relevance)
    return np.array(f).max()

def break_even_point(rank, relevance):
    precision = np.array(get_precision(rank, relevance))
    recall = np.array(get_recall(rank, relevance))
    index = np.where(precision == recall)
    return index, precision[index]

if __name__ == '__main__':
    targetDir = 'database_recordings'
    queryDir = 'query_recordings'
    fingerprintDir = 'fingerprint'
    output_file = 'output.txt'

    fingerprintBuilder(targetDir, fingerprintDir)
    audioIdentification(queryDir, fingerprintDir, output_file)

