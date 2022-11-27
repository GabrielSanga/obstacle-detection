#region LIBRARY

#Bibliotecas base
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import time
import os

# Importing Image module from PIL package
from PIL import Image
import PIL

#Transformação
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import RepeatedKFold
from PIL import Image
import cv2

#Classificador
from sklearn.svm import SVC

#Métricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

#endregion

app = Flask(__name__)

#region VARIABLE GLOBAL

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

EXTENSAO_PERMITIDA = set(['png', 'jpg', 'jpeg'])

PREDICT_PATH = 'C:/Users/gabri/TCC/predicts'
DATASET_PATH = "C:/Users/gabri/TCC/via-dataset-master/images"
FEATURE_PATH = "C:/Users/gabri/TCC/features/features.csv"
RESULT_PATH = "C:/Users/gabri/TCC/details-results/"

data_filename = RESULT_PATH+"data_detailed.csv"

# Criando folds para cross-validation - 10fold
kfold_n_splits = 10
kfold_n_repeats = 1
kf = RepeatedKFold(n_splits=kfold_n_splits, n_repeats=kfold_n_repeats, random_state=SEED)

image_size = (224, 224)

#endregion

#region FUNCTION

def fileValid(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSAO_PERMITIDA
 
def load_data():
    filenames = os.listdir(DATASET_PATH)
    categories = []
    
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'clear':
            categories.append(1)
        else:
            categories.append(0)
    
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    return df

def load_feature():
    return pd.read_csv(FEATURE_PATH, sep=',', usecols=range(1, 50177))

def gen_dataset(features, labels, train, test):
    dataset_train = np.array(features[train])
    dataset_train_label = np.array(labels[train])
    
    dataset_test = np.array(features[test])
    dataset_test_label = np.array(labels[test])
    
    return dataset_train, dataset_train_label, dataset_test, dataset_test_label

def training(train_data, train_label, test_data, clf):
    #Treinando o modelo
    start = time.time()   
    clf = clf.fit(train_data, train_label)  
    end = time.time() 
    
    time_trainning = end-start
    
    #Testando o modelo
    start = time.time()  
    classification_result = clf.predict(test_data)       
    end = time.time()

    time_prediction = end-start
        
    return time_trainning, time_prediction, classification_result 

def feature_model_extract(df):
    time_start = time.time()
    
    features_VGG16 = extract_features(df, modelVGG16, preprocessing_function_VGG16)
        
    features_VGG19 = extract_features(df, modelVGG19, preprocessing_function_VGG19)

    #concatenate array features VGG16+VGG19
    features = np.hstack((features_VGG16, features_VGG19))
            
    time_end = time.time()
    
    time_feature_extration = time_end - time_start
    
    return features, time_feature_extration

def create_models_VGG():
    IMAGE_CHANNELS = 3
    POOLING = None

    from keras.applications.vgg16 import VGG16, preprocess_input 
    global modelVGG16
    modelVGG16 = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=(224, 224) + (IMAGE_CHANNELS,))

    global preprocessing_function_VGG16
    preprocessing_function_VGG16 = preprocess_input

    from keras.applications.vgg19 import VGG19, preprocess_input
    global modelVGG19
    modelVGG19 = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=(224, 224) + (IMAGE_CHANNELS,))

    global preprocessing_function_VGG19
    preprocessing_function_VGG19 = preprocess_input

    from keras.layers import Flatten
    from keras.models import Model
    
    #VGG16
    output = Flatten()(modelVGG16.layers[-1].output)   
    modelVGG16 = Model(inputs=modelVGG16.inputs, outputs=output)

    #VGG19
    output = Flatten()(modelVGG19.layers[-1].output)   
    modelVGG19 = Model(inputs=modelVGG19.inputs, outputs=output)
        
    return True

def extract_features(df, model, preprocessing_function):         
    datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )

    total = df.shape[0]
    batch_size = 4
    
    generator = datagen.flow_from_dataframe(
        df, 
        PREDICT_PATH, 
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=False
    )
    
    features = model.predict(generator, steps=np.ceil(total/batch_size))

    return features

def classification(ds_features):
    return clf.predict(ds_features) 

#endregion
 
#region ROUTE

@app.route('/')
def main():
    df_feature = load_feature()
    df_data = load_data()

    #Carregando Labels
    labels = df_data["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)

    kf.split(df_data)

    all_feature = df_feature.to_numpy()

    #Instanciando classificador
    global clf
    clf = SVC(kernel="linear", C=0.025)

    #kfold loop
    for index, [train, test] in enumerate(kf.split(df_data)):     
        #Gerando os datasets
        dataset_train, dataset_train_label, dataset_test, dataset_test_label = gen_dataset(all_feature, labels, train, test)

        #Treinando o modelo
        time_trainning, time_prediction, pred = training(dataset_train, dataset_train_label, dataset_test, clf)
    
        hidden_labels = dataset_test_label.copy()
        hidden_pred = pred.copy()
 
        #csv detailed data
        with open(data_filename,"a+") as f_data:
            f_data.write("VGG16+VGG19,") #CNN
            f_data.write("LinearSVM,") #Classificador
            f_data.write(str(index+1)+",") #Kfold index
            f_data.write(str(np.shape(all_feature)[1])+"," ) #CNN_features
            f_data.write(str(0)+", ") 
            f_data.write(str("{0:.4f}".format(accuracy_score(hidden_labels,hidden_pred)))+",") #Acc Score
            f_data.write(str("{0:.4f}".format(f1_score(hidden_labels,hidden_pred)))+",") #F1 Score
            f_data.write(str("{0:.4f}".format(roc_auc_score(hidden_labels,hidden_pred)))+",") #ROC Score
            f_data.write(str("{0:.4f}".format(time_trainning))+",") #Time Classifier Trainning
            f_data.write(str("{0:.4f}".format(time_prediction))+",\n") #Time Classifier Predict

    #Criando e Treinando as CNN para extrair as features
    create_models_VGG()

    return "Classificador Treinado"
    
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        resp = jsonify({'message' : 'Imagem não encontrada na solicitação'})
        resp.status_code = 400
        return resp
 
    try:
        file = request.files['file']

        if file and fileValid(file.filename):            
            #Guardando a imagem em um dataframe
            df = pd.DataFrame({'filename': file.filename, "category": ["clear"]})
    
            filename = secure_filename(file.filename)
        
            # criando um objeto da imagem
            image = Image.open(file)
            
            # Girando a imagem 270 graus no sentido anti-horário
            image = image.rotate(270, PIL.Image.NEAREST, expand=1)

            # Redimensionando a imagem
            image = image.resize((700, 1000), Image.ANTIALIAS)

            #Salva a imagem
            image.save(os.path.join(PREDICT_PATH, filename + str(time.time())))
            
            #Extraindo as caracteristicas da imagem
            features, time_feature_extration = feature_model_extract(df)

            result = classification(features)

            #Renomeia o arquivo de acordo com o resultado obtido (SOMENTE PARA VALIDAÇÃO)
            if 1 == 2:
                if result == 1:
                    sNomeArquivo = "Clear" + str(time.time())
                else:
                    sNomeArquivo = "NoClear" + str(time.time())

                sNomeAntigo = os.path.join(PREDICT_PATH, filename)
                sNomeRenomeado = os.path.join(PREDICT_PATH, sNomeArquivo  + "." + filename.rsplit('.', 1)[1].lower())

                os.rename(sNomeAntigo, sNomeRenomeado)

            resp = jsonify({'result' : str(result[0])})
            resp.status_code = 201
            return resp

    except:
        resp = jsonify({"result", 0})
        resp.status_code = 500
        return resp
           
#endregion

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)