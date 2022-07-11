# Author: Dekun Xie #

import os 
import IPython.display as ipd
import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
import pickle

targets = []

def fingerprintBuilder(data_path, fingerprint_path):
    '''Generate fingerprint of the target databases.
        If the fingerprint exists, it will load that directly instead of generating'''
    global targets
    targetDir = data_path
    fingerprintDir = fingerprint_path      

    targets  = get_files(targetDir)

    coordinates_targets = []
    inverted_lists = []

    try:
        with open(os.path.join(fingerprint_path,'coordinates_targets.data'), 'rb') as f:
            coordinates_targets = pickle.load(f)

        with open(os.path.join(fingerprint_path, 'inverted_lists.data'), 'rb') as f:
            inverted_lists = pickle.load(f)

        print(f'Load existed fingerprints(coordinates_targets.data, inverted_lists.data) in folder: {fingerprint_path}')

    except BaseException as e:
        for i, target in enumerate(targets):
            stft = calculate_stft(target)
            coordinates_targets.append(calculate_contellation_map(stft,min_distance=5))
            inverted_lists.append(get_inverted_list(coordinates_targets[i]))

            print(f'Generating fingerprints {i+1}/{len(targets)}...')

        with open(os.path.join(fingerprintDir,'coordinates_targets.data'), 'wb') as f:
            pickle.dump(coordinates_targets, f)

        with open(os.path.join(fingerprintDir,'inverted_lists.data'), 'wb') as f:
            pickle.dump(inverted_lists, f)



def audioIdentification(query_path, fingerprint_path, output_path):
    '''Generate fingerprint of the query data to match.
    If the fingerprint exists, it will load that directly instead of generating.
    Also, the idetification results will be produced in .txt file'''
    global targets
    queryDir = query_path
    fingerprintDir = fingerprint_path
    queries = get_files(queryDir)
    coordinates_targets = []
    inverted_lists = []
    query_list = []
    coordinates_query = []

    with open(os.path.join(fingerprint_path,'coordinates_targets.data'), 'rb') as f:
        coordinates_targets = pickle.load(f)

    with open(os.path.join(fingerprint_path, 'inverted_lists.data'), 'rb') as f:
        inverted_lists = pickle.load(f)

    for i, q in enumerate(queries):
        stft = calculate_stft(q)
        coordinates_query.append(calculate_contellation_map(stft,min_distance=5))
        query_list.append(get_query_list(coordinates_query[i]))
        print(f'Generating query fingerprints {i+1}/{len(queries)}...')

    # test output.txt
    output_file = output_path
    with open(output_file,'w+') as f:
        content = ''
        f.writelines(content)

    for i in range(len(queries)):
        results = []
        query = get_file_name(queries[i])
        print(f'Matching {query}({i+1}/{len(queries)})...')
        for j in range(len(targets)):
            results.append(indicator_function(coordinates_targets[j], 
                                            coordinates_query[i], inverted_lists[j], query_list[i]))
        rank = get_rank(results)[0:3]
        print(f'Matched audio:{rank}')

        with open(output_file,'a') as f:
            content = f'{query}.wav\t{rank[0][0]}.wav\t{rank[1][0]}.wav\t{rank[2][0]}.wav\n'
            f.writelines(content)
            print(f'Written to {output_file}\n')

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

def get_answers(queries):
    answers = dict()
    for query in queries:
        query_name = get_file_name(query)
        amswer = query_name.split('-snippet')[0]
        answers.update({query_name: amswer})    
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

def get_inverted_list(coordinates):
    inverted_list = [0] * coordinates.shape[0]
    for i in range(coordinates.shape[0]):
        inverted_list[i] = np.where(coordinates[i]==True)[0]

    return inverted_list

def get_query_list(coordinates):
    query_list = np.dstack(np.where(coordinates.T==True))[0]
    return query_list

def indicator_function(coordinates_original, coordinates_query, inverted_list, query_list, top=5):
    indicator_max = coordinates_original.shape[1] + coordinates_query.shape[1]
    indicator_min = - coordinates_query.shape[1]
    
    key =[i for i in range(indicator_min, indicator_max)]
    indicator = dict.fromkeys(key)
    for k in indicator.keys():
        indicator[k] = 0
    
    for n, h in query_list:
        for i in inverted_list[h] - n:
            if i not in indicator.keys():
                increase = 0
            else:
                increase = indicator[i] + 1
            indicator.update({i:increase})

    result = sorted(indicator.items(), key=lambda x:x[1], reverse=True)[0:top]
    return result


def get_rank(results):
    rank = dict()
    for i, result in enumerate(results):
        rank.update({get_file_name(targets[i]): result[0][1]})
    
    return sorted(rank.items(), key=lambda x:x[1], reverse=True)

if __name__ == '__main__':
    targetDir = 'database_recordings'
    queryDir = 'query_recordings'
    fingerprintDir = 'fingerprint'
    output_file = 'output.txt'

    fingerprintBuilder(targetDir, fingerprintDir)
    audioIdentification(queryDir, fingerprintDir, output_file)
