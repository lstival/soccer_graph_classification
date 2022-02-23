import pandas as pd
import numpy as np

# from keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

import os

import joblib
import shap

# Lendo Imagens
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tqdm


class SVM_Wrapper:
    def __init__(self, svm_pickle):
        # Gerando transfer learning
        self.feature_extractor = DenseNet121(include_top=False, weights='imagenet', pooling='avg')
        self.svm = joblib.load(svm_pickle)

    def predict(self, tensor):
        features = self.feature_extractor.predict(tensor)
        return self.svm.predict(features)

    def predict_proba(self, tensor):
        features = self.feature_extractor.predict(tensor)
        return self.svm.predict_proba(features)


def f(X):
    tmp = X.copy()
    tmp = preprocess_input(tmp)
    return wrapper.predict_proba(tmp)


# Pega todas as imagens
def get_file_names(file):
    imagens = []
    for picture in os.listdir(file):
        imagens.append(img_to_array(load_img(file + '/' + picture, target_size=(256, 256))))
    return imagens


# Modelos que serão avaliados
# trained_models = ['mlp', 'sgdc', 'xgboost', 'gpc', 'svm']

# trained_models = ['xgboost', 'nearest', 'florest', 'sgdc']
trained_models = ['sgdc']

# Folde para salvar os resultados
path_save = 'shap_value_all_samples/'

pbar = tqdm.tqdm(trained_models)
for model_name in pbar:

    # Mensagem exbida durante o treinamento
    pbar.set_description(f"Treinando {model_name}")

    # Nome das classes que podem ser classificadas
    class_names = ['failure', 'success']

    # Criando modelo wrapper
    directory = 'C:/soccer_graph/modelos_/grouped_ad_fold_manual/9/'
    wrapper = SVM_Wrapper(directory + f"{model_name}_best_imagem_ad_grouped_ad_fold.sav")

    # Local onde estão as imagens
    path = 'c:/soccer_graph/rhythm_figures/sepair_grouped_image_ad_fold/train/9/1/'

    # Pegando os nomes das imagens
    imagens = get_file_names(path)

    # Todas as imagens para serem analisadas pelo SHAP
    x_all = np.array(imagens)

    # define a masker that is used to mask out partitions of the input image, this one uses a blurred background
    masker = shap.maskers.Image("inpaint_telea", x_all[0].shape)

    # By default the Partition explainer is used for all  partition explainer
    explainer = shap.Explainer(f, masker, output_names=class_names)

    # here we use 1000 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(x_all, max_evals=1000, batch_size=2, outputs=shap.Explanation.argsort.flip[:1])

    df_shap_values = pd.DataFrame()
    list_shap_values = []

    for sample in range(len(x_all)):
        shap_array = []
        for i in range(len(shap_values.values[:, :, 0, :][sample])):
            shap_array.append(float(shap_values.values[:, :, 0, :][sample][i][0]))
        list_shap_values.append(shap_array)
        # df_shap_values[sample] = shap_array

    # using the savetxt to save list of values calculates

    os.makedirs(os.path.dirname(path_save), exist_ok=True)

    np.savetxt(f"{path_save}/shap_values_all_{model_name}.csv",
               list_shap_values,
               delimiter=", ",
               fmt='% s')

