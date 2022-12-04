#Arquivo responsável por pegar as imagens recebidas do APP (já conferidas) e gerar as métricas

#base libraries
import pandas as pd
import os

#Métricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "C:/Users/gabri/TCC/predicts/"
RESULT_PATH = "C:/Users/gabri/TCC/details-results/results_predict/"

#----------------------- FUNCTION ------------------------------------------------

def load_data():
    filenames = os.listdir(DATASET_PATH)
    aLabel = []
    aPredict = []
    
    for filename in filenames:
        #Carregando os labels
        label = filename.split('.')[3]
        if label == 'clear':
            aLabel.append(1)
        else:
            aLabel.append(0)


        predict = filename.split('.')[0]
        if predict == 'clear':
            aPredict.append(1)
        else:
            aPredict.append(0)
    
    df = pd.DataFrame({
        'label': aLabel,
        'predict': aPredict
    })

    return df

#----------------------- MAIN ------------------------------------------------

#Carregando as imagens em um dataframe
df = load_data()

hidden_labels = df["label"].to_numpy()
hidden_pred = df["predict"].to_numpy()

#Gerando as informações
with open(RESULT_PATH + "data_detailed.csv", "a+") as f_data:
    f_data.write("VGG16+VGG19,")
    f_data.write("LinearSVM,")
    f_data.write(str(0)+", ") 
    f_data.write(str("{0:.4f}".format(accuracy_score(hidden_labels,hidden_pred)))+",") #Acc Score
    f_data.write(str("{0:.4f}".format(f1_score(hidden_labels,hidden_pred)))+",") #F1 Score
    f_data.write(str("{0:.4f}".format(roc_auc_score(hidden_labels,hidden_pred)))+",") #ROC Score

#Gerando a Matrix de Confusão
cm = confusion_matrix(hidden_labels, hidden_pred)
sns.heatmap(cm, annot=True)

#Plotando a figura, salvando e limpando
plt.plot(cm)
plt.savefig(RESULT_PATH + "matriz_confusao.png", type="png", dpi=100)
plt.clf()