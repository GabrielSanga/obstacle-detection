#base libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import time
import os

#transformation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

DATASET_PATH = "C:/Users/gabri/TCC/via-dataset-master/images/"
RESULT_PATH = "C:/Users/gabri/TCC/results/.features.csv"

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

def feature_model_extract():   
    start = time.time()
            
    model_type = 'VGG16'
    modelVGG16, preprocessing_functionVGG16, image_sizeVGG16 = create_model(model_type)
    features_VGG16 = extract_features(df, modelVGG16, preprocessing_functionVGG16, image_sizeVGG16)
        
    model_type = 'VGG19'
    modelVGG19, preprocessing_functionVGG19, image_sizeVGG19 = create_model(model_type)
    features_VGG19 = extract_features(df, modelVGG19, preprocessing_functionVGG19, image_sizeVGG19)
    
    #concatenate array features VGG16+VGG19
    features = np.hstack((features_VGG16,features_VGG19))
        
    end = time.time()
    
    time_feature_extration = end-start
    
    return features, time_feature_extration

def create_model(model_type):   
    #CNN Parameters
    IMAGE_CHANNELS=3
    POOLING = None # None, 'avg', 'max'
    
    # load model and preprocessing_function
    if model_type=='VGG16':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input   
        model = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
        
    elif model_type=='VGG19':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,)) 
        
    else: print("Error: Model not implemented.")

    preprocessing_function = preprocess_input

    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model
    
    output = Flatten()(model.layers[-1].output)   
    model = Model(inputs=model.inputs, outputs=output)
        
    return model, preprocessing_function, image_size

def extract_features(df, model, preprocessing_function, image_size):
    df["category"] = df["category"].replace({1: 'clear', 0: 'non-clear'}) 
           
    datagen = ImageDataGenerator(
        #rescale=1./255,
        preprocessing_function=preprocessing_function
    )
    
    total = df.shape[0]
    batch_size = 4
    
    generator = datagen.flow_from_dataframe(
        df, 
        DATASET_PATH, 
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    features = model.predict(generator, steps=np.ceil(total/batch_size))
    
    return features

#----------------------- MAIN ------------------------------------------------
#Carregando as imagens em um dataframe
df = load_data()

#Extraindo as caracteristicas das imagens
features, time_feature_extration = feature_model_extract()

#Convertendo as caracteristicas em uma dataframe
df_csv = pd.DataFrame(features)
 
#Salvando o dataframe em uma arquivo csv
df_csv.to_csv(RESULT_PATH)