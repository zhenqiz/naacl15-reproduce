#   Read-in Word Embedding
import numpy as np
from numpy import dot
from numpy.linalg import norm
from multiprocessing import Process
import os

import heapq

import random

def Fe(a, b, type = 0):
    #   Cos Similarity
    if type == 0:
        return dot(a, b) / (norm(a) * norm(b))
    #   Rank

def extract_directions(candid_stags, stags, voc):
    print('Child process is running %s ...' % (os.getpid()))
    #   Set threshold
    t0_rank = 100
    t_rank = 30
    t_cos = 0.5
    #   Find rules, in our cases, rules are just like
    #   wordt123 -> wordt234
    # for (stag, word) in stags.items():
    #     if len(word) > 50:
    #         max_len = len(word)
    #         print(stag)
    #         print(word)
    #         print(len(word))

    #   Downsample the word pairs
    max_L = 1000
    all_directions = []
    for (stag1, words1) in candid_stags.items():
        for (stag2, words2) in stags.items():
            if stag1 == stag2:
                continue
            # print(stag1 + " " + stag2)
            #   rule: stag1->stag2
            word_pairs = []

            for w1 in words1:
                for w2 in words2:
                    word_pairs.append((w1, w2))

            if len(word_pairs) > max_L:
                word_pairs = random.sample(word_pairs, max_L)

            L = len(word_pairs)
            # print("The size of Sr:" + str(L))
            # S = [i for i in range(L)]
            #   Store the result to avoid redundant computation
            # CosE = np.zeros((L, L))
            # RankE = np.zeros((L, L))

            #   Find top t0_rank(100) (w1,w2)X(w3,w4) pair
            #   Maintain a heap, each entry = (CosE[i,j], i*L+j, (w1, w2) = dw, (w3, w4))
            heap = []
            heapq.heapify(heap)
            for i, Wi in enumerate(word_pairs):
                    #   dw is the direction vector
                    (w1, w2) = Wi
                    dw = voc[w2] - voc[w1]
                    for j, Wj in enumerate(word_pairs):
                        if i == j:
                            continue
                        (w3, w4) = Wj
                        # CosE[i, j] = Fe(voc[w4], voc[w3] + dw)
                        heapq.heappush(heap, (Fe(voc[w4], voc[w3] + dw), i*L + j, Wi, Wj))
                        if len(heap) > t0_rank:
                            heapq.heappop(heap)

            #   Find best directions, delete pairs that it covers and keep finding the next best
            best_directions = []
            d_threshold = 3
            while 1:
                max_dw = 0
                D = {}
                #   Find the best direction for current S
                for (_, _, (w1, w2), _) in heap:
                    if w1+w2 not in D:
                        D[w1 + w2] = 1
                    else:
                        D[w1 + w2] += 1
                        if D[w1 + w2] > max_dw:
                            max_dw = D[w1 + w2]
                            best_dw = (w1, w2)

                #   If the number of covered pairs less than threshold, abondon and exit
                if max_dw < d_threshold:
                    break
                #   Append Best direction to the direction list
                best_directions.append(best_dw)
                #   S-S_w0 = those uncovered pairs
                heap = [(cosE, index, Wi, Wj) for (cosE, index, Wi, Wj) in heap if Wi != best_dw]

            all_directions.append(best_directions)

    output_path = 'output_direction_' + str(os.getpid()) + '.txt'
    with open(output_path, 'w') as data_file:
        for _, best_directions in enumerate(all_directions):
            for j, key in enumerate(best_directions):
                data_file.write(key[0] + " " + key[1])
                if j == len(best_directions) - 1:
                    data_file.write('\n')
                else:
                    data_file.write(' ')

if __name__=='__main__':
    random.seed(1)
    voc = {}
    voc_index = {}
    glove_path = 'vectors_threshold5.txt'
    '''
        Glove Word Embedding File explain:
        Each line start with a word and its embedding vector.
    '''
    i = 0
    with open(glove_path, "r") as data_file:
        for line in data_file:
            line = line.strip("\n")

            # skip blank lines
            if not line:
                continue

            line_list = line.split()

            word = line_list[0]

            voc[word] = np.array([float(line_list[i]) for i in range(1, len(line_list))])
            voc_index[word] = i
    #   Finding rules
    voc_size = len(voc)
    #   We only consider the change of suertag suffix in this case
    #   So first we need to extract all possible supertags
    stags = {}
    for w in voc:
        stag_tmp = ''
        for i in range(len(w) - 1, 0, -1):
            stag_tmp = w[i] + stag_tmp
            if w[i] == 't':
                if stag_tmp not in stags:
                    stags[stag_tmp] = set()
                stags[stag_tmp].add(w)
                break

    #   Parallze the work
    candid_stags_p1 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 0}
    candid_stags_p2 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 1}
    candid_stags_p3 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 2}
    candid_stags_p4 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 3}
    candid_stags_p5 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 4}
    candid_stags_p6 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 5}
    candid_stags_p7 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 6}
    candid_stags_p8 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 7}
    candid_stags_p9 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 8}
    candid_stags_p10 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 9}
    candid_stags_p11 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 10}
    candid_stags_p12 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 11}
    candid_stags_p13 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 12}
    candid_stags_p14 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 13}
    candid_stags_p15 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 14}
    candid_stags_p16 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 15}
    candid_stags_p17 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 16}
    candid_stags_p18 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 17}
    candid_stags_p19 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 18}
    candid_stags_p20 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 19}
    candid_stags_p21 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 20}
    candid_stags_p22 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 21}
    candid_stags_p23 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 22}
    candid_stags_p24 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 23}
    # candid_stags_p25 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 24}
    # candid_stags_p26 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 25}
    # candid_stags_p27 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 26}
    # candid_stags_p28 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 27}
    # candid_stags_p29 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 28}
    # candid_stags_p30 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 29}
    # candid_stags_p31 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 30}
    # candid_stags_p32 = {key: stags[key] for i, key in enumerate(stags) if i % 24 == 31}
    p1 = Process(target=extract_directions, args=(candid_stags_p1, stags, voc))
    p2 = Process(target=extract_directions, args=(candid_stags_p2, stags, voc))
    p3 = Process(target=extract_directions, args=(candid_stags_p3, stags, voc))
    p4 = Process(target=extract_directions, args=(candid_stags_p4, stags, voc))
    p5 = Process(target=extract_directions, args=(candid_stags_p5, stags, voc))
    p6 = Process(target=extract_directions, args=(candid_stags_p6, stags, voc))
    p7 = Process(target=extract_directions, args=(candid_stags_p7, stags, voc))
    p8 = Process(target=extract_directions, args=(candid_stags_p8, stags, voc))
    p9 = Process(target=extract_directions, args=(candid_stags_p9, stags, voc))
    p10 = Process(target=extract_directions, args=(candid_stags_p10, stags, voc))
    p11 = Process(target=extract_directions, args=(candid_stags_p11, stags, voc))
    p12 = Process(target=extract_directions, args=(candid_stags_p12, stags, voc))
    p13 = Process(target=extract_directions, args=(candid_stags_p13, stags, voc))
    p14 = Process(target=extract_directions, args=(candid_stags_p14, stags, voc))
    p15 = Process(target=extract_directions, args=(candid_stags_p15, stags, voc))
    p16 = Process(target=extract_directions, args=(candid_stags_p16, stags, voc))
    p17 = Process(target=extract_directions, args=(candid_stags_p17, stags, voc))
    p18 = Process(target=extract_directions, args=(candid_stags_p18, stags, voc))
    p19 = Process(target=extract_directions, args=(candid_stags_p19, stags, voc))
    p20 = Process(target=extract_directions, args=(candid_stags_p20, stags, voc))
    p21 = Process(target=extract_directions, args=(candid_stags_p21, stags, voc))
    p22 = Process(target=extract_directions, args=(candid_stags_p22, stags, voc))
    p23 = Process(target=extract_directions, args=(candid_stags_p23, stags, voc))
    p24 = Process(target=extract_directions, args=(candid_stags_p24, stags, voc))
    # p25 = Process(target=extract_directions, args=(candid_stags_p25, stags, voc))
    # p26 = Process(target=extract_directions, args=(candid_stags_p26, stags, voc))
    # p27 = Process(target=extract_directions, args=(candid_stags_p27, stags, voc))
    # p28 = Process(target=extract_directions, args=(candid_stags_p28, stags, voc))
    # p29 = Process(target=extract_directions, args=(candid_stags_p29, stags, voc))
    # p30 = Process(target=extract_directions, args=(candid_stags_p30, stags, voc))
    # p31 = Process(target=extract_directions, args=(candid_stags_p31, stags, voc))
    # p32 = Process(target=extract_directions, args=(candid_stags_p32, stags, voc))

    print('Process will start.')
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p8.start()
    p6.start()
    p7.start()
    p9.start()
    p10.start()
    p11.start()
    p12.start()
    p13.start()
    p14.start()
    p15.start()
    p16.start()
    p17.start()
    p18.start()
    p19.start()
    p20.start()
    p21.start()
    p22.start()
    p23.start()
    p24.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p8.join()
    p6.join()
    p7.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()
    p13.join()
    p14.join()
    p15.join()
    p16.join()
    p17.join()
    p18.join()
    p19.join()
    p20.join()
    p21.join()
    p22.join()
    p23.join()
    p24.join()
    print('Process end.')
#   For each pair of word, evaluate them with each rule
#   Using two criteria, Cos and rank




