#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import collections
import warnings
from scipy import stats

#%%
# Leitura do csv
# , columns=['peso', 'altura']
df = pd.read_csv("dados/PM251.csv", sep=',', usecols=['year','month','day','hour','DEWP','TEMP','PRES','Iws','Is','Ir'])
df.head(5)

#%%
# Correlação das colunas
df.corr(method ='pearson') 

#%%

def pearsonr_ci(x,y,alpha=0.05):
   r, p = stats.pearsonr(x, y)
   r_z = np.arctanh(r)
   se = 1 / np.sqrt(x.size - 3)
   z = stats.norm.ppf(1 - alpha / 2)
   lo_z, hi_z = r_z - z * se, r_z + z * se
   lo, hi = np.tanh((lo_z, hi_z))
   return r, p, lo, hi

#%%

def checa_correlacao(r, p, lo, hi):
    if (r > lo) & (r < hi) :
        #if (r == base):
        val_valida = valida_forca_correlacao(r, lo, hi)
        return r, lo, hi, val_valida
    return 0, 0, 0, 0
#%%

#Funcao verifica a força da correlação entre os campos
def valida_forca_correlacao(r, lo, hi): 
    neutro = (lo+ ((hi-lo) / 2))
    parte = (neutro - lo) / 3
    menor = lo
    menor2 = menor + parte
    menor1 = menor2 + parte
    neutro =  menor1 + parte
    maior1 = neutro + parte
    maior2 =  maior1 + parte
    maior = hi
    
    if(r >= menor) & (r < menor2):      
        return "RELAÇÃO FORTE NEGATIVA"
    elif(r >= menor2) & (r < neutro):  
        return "RELAÇÃO FRACA NEGATIVA"
    elif(r <= maior) & (r >= maior2):  
        return "RELAÇÃO FORTE POSITIVA"
    elif(r < maior2) & (r >= neutro):  
        return "RELAÇÃO FRACA POSITIVA"
        
#%%
c1 = 0
for i, j in df.iteritems():
    c2 = 0
    for i2, j2 in df.iteritems():
        r, p, lo, hi = pearsonr_ci(df[i], df[i2])
        val_r, val_lo, val_hi, val_valida = checa_correlacao(r, p, lo, hi)
        if(val_r > 0) & (i != i2):
            print(i, "/", i2," => ", val_r, val_lo, val_hi, val_valida)
        c2 += 1        
    c1 += 1    
    
#%%

# Normalização
# , columns=['year', 'month','day','hour','DEWP','TEMP','PRES','Iws','Is','Ir'] 
df = pd.read_csv("dados/PM251.csv", sep=',')
normal = pd.get_dummies(df)
print(normal.head(5))
filename = 'dados/PM251_Normal.csv'
normal.to_csv(path_or_buf=filename)

#%%
#Gerar os modelos
filename = 'Models/PM251_Normal.sav'
kmeans = KMeans(n_clusters=4).fit(normal)
centroids = kmeans.cluster_centers_
pickle.dump(kmeans, open(filename, 'wb'))

#%%
#Carregar os modelos
kmeans = pickle.load(open(filename, 'rb'))
result = kmeans.predict([
    [2010,1,2,0,-16,-4,1020.0,1.79,0,0,0,0,1,0,1,0,0,0],
    [2014,1,2,0,-10,-5,120.0,1.05,0,0,0,0,1,0,0,1,0,0]
    # [2010,1,2,0,-16,-4,10200,179,0,0,0,0,1,0,1,0,0,0],
    # [2011,1,2,1,-15,-4,10200,268,0,0,0,0,1,0,0,0,0,1],
    # [2012,1,2,2,-11,-5,10210,357,0,0,0,0,1,0,0,0,0,1],
    # [2013,1,2,3,-7,-5,10220,536,1,0,0,0,1,0,0,0,0,1],
    # [2014,1,2,0,-10,-5,120,0,1.05,0,0,0,1,0,1,1,1,1],
    # [2013,1,2,5,-7,-6,10220,714,3,0,0,0,1,0,1,0,0,0],
    # [2012,1,2,6,-7,-6,10230,893,4,0,0,0,1,0,1,1,1,1],
    # [2011,1,2,7,-7,-5,10240,1072,0,0,0,0,1,0,1,0,0,0]
])

# O resultado é uma instância que será inferida.
# O resultado representa o cluster ao qual essa instância pertence.
print(result)


#%%
