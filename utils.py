#Utils com métodos utilizados pelos noteboks
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

def other_games_index():
    half1 = [499, 424, 288, 234, 496, 110,  39, 375, 419, 435, 180, 221, 327,
            318, 220, 399, 189, 368, 274,  70, 456, 356, 229, 175,  87, 486,
            31, 444, 235, 290, 283, 190, 159, 155,  34, 191, 493,  32, 420,
            317, 181, 111, 251, 360, 286, 403, 135,  18, 231, 411, 314, 371,
            243, 459, 207, 416, 417, 129, 188, 106, 107,  74, 297, 227, 491,
            61, 174, 350, 311, 239,  72,  12, 430,  56,  66, 365, 122, 421,
            489, 469, 128, 326, 169, 257,   2, 247, 316, 204,  45, 102, 264,
            394, 165, 152, 145, 236, 133, 208, 113, 478, 348, 437, 384, 450,
            344,  23, 230, 331, 104,  47, 343,  68, 192, 482,  60,   8, 292,
            377, 337, 100, 282, 425, 306, 151, 334,  36, 103, 258, 349, 328,
            194, 373, 109,  64, 398, 341, 391, 467,  75, 298, 162,   1, 266,
            26,  84, 284, 401,  51,  27, 307, 216, 105, 206, 427,   5,  57,
            462, 300,   9,  28, 372, 261,  88, 140, 146,  83, 324,  35,  16,
            108, 442, 276, 154, 475, 281, 461,  15, 271, 465,  49, 242, 423,
            397,  77, 441,  97,  52, 244, 346, 213, 473, 279, 406, 184, 259,
            116,  99, 332, 130, 415,  25, 404, 357, 299, 466, 388, 185, 374,
            215,  58, 173, 455, 254, 177,  81,  95, 304, 248,   4, 313,  22,
            490, 429, 457, 289, 287,  73, 219, 202,  50, 474,   3, 333,  94,
            390, 237,  40, 183, 163, 460, 366, 197, 498, 497,  43, 345, 477,
            353,  19, 218]

    return np.array(half1)

def metrics(polygons=False,complet=True):
    if polygons == True:
        METRICS = ['grapf_centroid','grapf_aerea']
    else:
        METRICS = ['richclubcoefficient','averdist','betcen','clusteringcoeffi','degree','eccentricity','entropy',
        'globalefficiency','locefficiency','pagerank','vulnerability']

    if complet == False:
        METRICS = ['betcen','clusteringcoeffi','eccentricity','entropy',
        'globalefficiency','locefficiency','pagerank','vulnerability']

    return METRICS

def partidas():
    partidas = ['REDSCAT1', 'REDSCAT2', 'REDSAJT1', 'REDSAJT2', 'REDMACT1',
       'REDMACT2', 'REDVELT1', 'REDVELT2', 'CapBotT1', 'CapBotT2', 'CapCorT1', 'CapCorT2', 'CapPalT1',
       'CapPalT2', 'CapSpoT1', 'CapSpoT2', 'sanituT1', 'sanituT2',
       'SpoFlaT1', 'SpoFlaT2']

    return partidas


def game_test(alterned=False, half=1):
    game_names =    ['CapBotT1', 'CapBotT2', 'CapCorT1', 'CapCorT2', 'CapPalT1',
    'CapPalT2', 'CapSpoT1', 'CapSpoT2', 'sanituT1', 'sanituT2',
    'SpoFlaT1', 'SpoFlaT2']

    game_names_alterned1 =  [['CapBotT1'], ['CapCorT2'], ['CapPalT1'],
    ['CapSpoT2'], ['sanituT1'], ['SpoFlaT2']]

    game_names_alterned2 = ['CapBotT2', 'CapCorT1', 
    'CapPalT2', 'CapSpoT1', 'SpoFlaT1', 'sanituT2']

    if alterned != True:
        return game_names
    if half == 1:
        return game_names_alterned1 
    return game_names_alterned2

def get_file_names(file,metric,folder):
    imagens = []
    for picture in os.listdir(folder+file):
        if picture.split("_")[2] == str(metric):#Verificando se é a métrica desejada
            imagens.append(img_to_array(load_img(folder+file+'/'+picture)))
    return imagens

def image_to_array(imagens):
    concat = []
    for imagem_idx in range(len(imagens)):
        if imagem_idx == 0:
            concat = imagens[imagem_idx]
        else:
            concat = np.concatenate((concat,imagens[imagem_idx]))
    return concat

def read_files_on_fold(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles

#Lê os arquivos de métricas de grafos 
def read_dataframe_metric(file,mypath):
    #metric = pd.read_csv('grafo//'+ partida+ '/'+graph_metrics[metric_count]+'.csv')
    metric = pd.read_csv(mypath+'/'+str(file))
    metric.drop(metric.columns[[0]],1, inplace= True)
    metric.fillna(0,inplace= True)
    
    return metric