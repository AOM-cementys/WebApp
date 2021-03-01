#script streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit_functions as sf
#%matplotlib inline

import scipy as sp
from scipy import signal

import pandas as pd
from sklearn import preprocessing
pd.options.mode.chained_assignment = None
from pathlib import Path
import os
import re

def general_concat(list_data):

    """
    La fonction general_concat prend en entrée les fichiers deposés sur la page.Cad une liste de fichiers txt
    et xls comprenant un channel et sa pression associée à une Température donnée.

    elle trie ensuite les fichiers à l'aide de la fonction trait_list_file. Cette fonction prend une liste 
    en entrée et renvoie une liste du tuple contenant les couples channel,pression pour une temperature donnée

    elle applique la fonction process_traitement à tous les couples présents dans la liste de couples.
    Puis concatene les dataframe issues de l'application process_traitement.
    Elle renvoir le dataframe final.
    """
    def takeName(elem):
        return elem.name
    list_data.sort(key=takeName)

    general_df = pd.DataFrame()
    list_couple = trait_list_file(list_data)
    list_df=[]
    list_df_NT = []
    list_weak_corr=[]
    list_temp=[]
    
    dict_new={}
    
    for line in list_couple:
        
        new_df_NT,mini_df, dict_decalage1, dict_total, temp = process_traitement(line[0],line[1])
        
        
        list_df.append(mini_df)
        list_df_NT.append(new_df_NT)
        dict_new[temp]=dict_total


    dict_df = pd.DataFrame(dict_new)

    
    general_df = pd.concat(list_df)
    general_df= general_df.reset_index().drop(columns='index')
    
    general_df_NT = pd.concat(list_df_NT)
    general_df_NT= general_df.reset_index().drop(columns='index')

    return general_df_NT,general_df,dict_df

def process_traitement(fichier_channel,fichier_pression):
    """
    La fonction process_traitement prend en entrée un fichier correspondant au fichier channel
    cad un fichier texte et un fichier pression en xls
    elle applique ensuite une succession de fonction :
    - une fonction df_P_T_Lambda_lies qui transforme les fichiers en dataframe
    - une fonction construction_df qui joint les deux fichiers en leur fixant des noms pour les colonnes
    - une fonction gaussian smooth pour lisser les données
    - une fonction dict-decalage1 qui effectue la correlation entre pression et lambdaP et lambdaT
      afin de déterminer le décalage existant entre lambdaP ou T et la pression. Ce décalage est donné
      pour la valeur de corrélation la plus grande
    - une fonction decalator qui vient appliquer ce decalage au lambdaP et T et renvoie un df augmenté de
      ces courbes decalées.
    - enfin une fonction Ndf_courbes_decalees qui prend en entrée le df crée précedemment par decalator
      Cette fonction fait appelle à la fonction borneur et à la fonction montee_palier_remover.
      borneur permet de déterminer les limites de la zone d'interet. C'est à dire de définir automatiquement
      en exploitant la dérivée de la Pression la zone que l'on conserve dans le df final. Cette opération 
      intervient après la corrélation afin de conserver le maximum de points pour effectuer celle-ci.
      Pour la même raison, la fonction montee_palier_remover intervient en final. Cette fonction permet
      d'enlever du dataframe les zones de passage entre palier de pression une fois encore en exploitant la dérivée
      de la Pression. Ces zones correpondent à des transitions rapides où on ne peut considérer que la temp
      -erature et la reponse du capteur sont bien stabilisées. 

    Une fois ces opérations effectuées, process_traitement renvoie un dataframe avec les courbes lambdaP et lambdaT
    recalées par rapport à la Pression pour une température donnée.  

    """


    df_channel, df_pression, temp = df_P_T_Lambda_lies(fichier_channel,fichier_pression)
    #on garde le dataframe brut non traité
    new_df_NT = construction_df(df_pression,df_channel)
    
   
    new_df = gaussian_smooth(new_df_NT)
    new_df = create_diff(new_df)
    dict_decalage1,dict_total= dict_decalage(np.abs(new_df))
    
    new_df = decalator(new_df,dict_decalage1)
    mini_df = Ndf_courbes_decalees(new_df)
    #mini_df  = mini_df.groupby(["Pression","Temperature"], as_index=False)[mini_df.columns].mean()
    mini_df,right_stop =remove_all_above(mini_df)
    #correlator_warning(mini_df)
    
    return new_df_NT,mini_df, dict_decalage1,dict_total, temp



def trait_list_file (list_files):
    def takeName(elem):
        
        return elem.name
    list_files.sort(key=takeName)
    list_couple=[]
    for ele in list_files:
        if ele.name.startswith("Pression") or ele.name.startswith("pression") or ele.name.startswith("Channel") or ele.name.startswith("channel"):
            if re.findall(r'\d+',ele.name)!=[]:
                temp = float(re.findall(r'\d+',ele.name)[-1])
                for ele2 in list_files:
                    if re.findall(r'\d+',ele2.name)!=[]:
                        if temp == float((re.findall(r'\d+',ele2.name)[-1])):
                            if ele!=ele2 :

                                    list_couple.append((ele,ele2))

    for i,ele in enumerate(list_couple):
        for j,ele in enumerate(list_couple):
            if i!=j:
                if list_couple[i][0]==list_couple[j][1]:
                    list_couple.remove(list_couple[j])
    return list_couple

def df_P_T_Lambda_lies (fichier_channel, fichier_pression):
    
    

    temp = float(re.findall(r'\d+',str(fichier_channel.name))[-1])
    
    length= len(pd.read_table(fichier_channel,delim_whitespace=True,error_bad_lines=False,warn_bad_lines=False).columns)
    try:
        fichier_channel.seek(0)
    except:
        1
    renommage_col=["col_name"+str(i) for i in range(0,length)]
    df_channel = pd.read_table(fichier_channel,decimal=",", delim_whitespace=True,error_bad_lines=False,names=renommage_col)
    
    

    df_pression = pd.read_excel(fichier_pression)
    df_pression["Temperature"] = temp
    return df_channel,df_pression, temp

#convertit un fichier channel en provenance de lambdasoft d'une frequence quelconque vers une frequence de 1hz pareille à la fréquence du capteur de pression.
def converter_fech1hz(df):
    def is_entier(ele):
        if type(ele)==float:
            return float(ele).is_integer()
    df_freq1 = df[df["col_name0"].apply(lambda x:is_entier(x))]
    return df_freq1

def construction_df(df_pression,df_channel):

    #fonction qui prend soit un fichier issu de lambda soft soit un fichier issu de bragsoft
    if (df_channel["col_name0"].iloc[0]).startswith("Time(s)"):
        df_lambdasoft=df_channel
        df_lambdasoft = df_lambdasoft.drop([0])
        df_lambdasoft  = df_lambdasoft.astype(float)
        df_freq1 = converter_fech1hz(df_lambdasoft)
        df_freq1.reset_index(drop=True, inplace=True)

        df_channel = df_freq1

        list_nom_colonnes = ["Time"]
        poubelle = []
        for i in range(0,len((df_channel.columns))-4,4):
            #print(i)

            index = i//4
            list_nom_colonnes.append("lambdaP"+str(index))
            list_nom_colonnes.append("P_Amp"+str(index))

            list_nom_colonnes.append("lambdaT"+str(index))
            list_nom_colonnes.append("T_Amp"+str(index))


            poubelle.append("P_Amp"+str(index))

            poubelle.append("T_Amp"+str(index))

        df_channel.columns = list_nom_colonnes
        del df_channel['Time']
        df_channel = df_channel.drop(columns=poubelle)
    else:
        list_nom_colonnes = ["date","hour"]
        poubelle = []
        for i in range(0,len((df_channel.columns))-6,6):
            #print(i)
            index = i//6
            list_nom_colonnes.append("lambdaP"+str(index))
            list_nom_colonnes.append("P_Amp"+str(index))
            list_nom_colonnes.append("C"+str(index))
            list_nom_colonnes.append("lambdaT"+str(index))
            list_nom_colonnes.append("T_Amp"+str(index))
            list_nom_colonnes.append("F"+str(index))

            poubelle.append("P_Amp"+str(index))
            poubelle.append("C"+str(index))
            poubelle.append("T_Amp"+str(index))
            poubelle.append("F"+str(index))


        df_channel.columns = list_nom_colonnes

        #for name_col in df_channel.columns[2:]:
                #df_channel[str(name_col)] = df_channel[str(name_col)].apply(lambda x: np.float(x.replace(",",".")))

        #df_channel["date"] = df_channel["date"]+str(" ")+df_channel["hour"]
        del df_channel['hour']
        del df_channel['date']
        #pd.to_datetime(df_channel['date'])

        df_channel = df_channel.drop(columns=poubelle)
    
    
    
    df_n = pd.concat([df_pression,df_channel],axis=1)
    list_colo_drop=["Datetime","Unit","Temperature(℃)","Interval(s)"]
    df_n = df_n.drop(columns=list_colo_drop)
    df_n =df_n.rename(columns={"Pressure":"Pression"})

    return df_n

def standardize_df_MinMax(df):
    df=(df-df.min())/(df.max()-df.min())
    return df

def standardize_df_std(df):
    df=(df-df.mean())/(df.std())
    return df

def gaussian_smooth(df):
    df.rolling(window=20, win_type='gaussian', center=True).mean(std=5)
    return df

def create_diff(df):
    df["diff_Pression"]=df["Pression"].diff(10)
    for ele in df.columns:
        if ele.startswith("lambdaP") or ele.startswith("lambdaT"):
            df["diff_"+str(ele)]=df[ele].diff(10)
    return df

def correlator_1(df,name_col1,name_col2):
    df = df.fillna(0)
    corr_list = [df[str(name_col1)].corr(df[str(name_col2)].shift(i)) for i in range(-len(df)//2,len(df)//2,1)]
    #corr_list = [0 if np.isnan(x) else x for x in xcov_monthly]


    corr_list = np.array(corr_list)
    corr_max = np.max(corr_list)
    tau = np.argmax(corr_list) - (len(df)//2)

    return corr_max,tau

def correlator_2(df,name_col1,name_col2):
    
    df=(df-df.min())/(df.max()-df.min())
    df = df.fillna(0)
    
    x= np.array(df[str(name_col1)] )
    y = np.array(df[str(name_col2)])
    corr = sp.signal.correlate(x,y)
    corr_max = np.max(corr)/len(df) #génére une valeur de corrélation peu interpretable
    tau = np.argmax(corr)- len(df)
    
    corr_max = df[str(name_col1)].corr(df[str(name_col2)].shift(tau))# on préfère la méthode de Pandas qui renvoie une valeur comprise entre 0 et 1

    return corr_max,tau

def dict_decalage (df) : 
    df=np.abs(df)
    df=standardize_df_MinMax(df)
    dict_decal={}
    dict_decal_tot={}
    weak_corr=[]
    for ele in df.columns:
        if ele.startswith('lambdaP') or ele.startswith('lambdaT'):
            if correlator_2(df,'diff_Pression',"diff_"+str(ele))[0] >0.6: 
                dict_decal[ele]=correlator_2(df,'diff_Pression',"diff_"+str(ele))
                dict_decal_tot[ele] = correlator_2(df,'diff_Pression',"diff_"+str(ele))[0]
            else :
                if ele.startswith("lambdaP"):
                    if dict_decal.get("lambdaT0") is not None:
                        dict_decal[ele]=dict_decal["lambdaP0"]
                    else:
                        dict_decal_tot[ele] = correlator_2(df,'diff_Pression',"diff_"+str(ele))[0]
                elif ele.startswith("lambdaT"):
                    if dict_decal.get("lambdaT0") is not None:
                        dict_decal[ele]=dict_decal["lambdaT0"]
                    else:
                        dict_decal_tot[ele] = correlator_2(df,'diff_Pression',"diff_"+str(ele))[0]
    
    return dict_decal,dict_decal_tot

def decalator (df, dict_decal):
    
    #Prend une df et un dictionnaire des decalages et retourne le df augmenté des colonnes décalées

    for ele in df.columns:
        if str(ele) in dict_decal:
            #print("decalage fait pour "+str(ele))
            tau = dict_decal[str(ele)][1]
            df["decale_"+str(ele)] = df[ele].shift(tau).ffill()            

        
        if str(ele) not in dict_decal:
        
        
            if  str(ele).replace("P","T") in dict_decal:
                tau = dict_decal[str(ele).replace("P","T")][1]
                df["decale_"+str(ele)] = df[ele].shift(tau).ffill() 
            
            if  str(ele).replace("T","P") in dict_decal:
                tau = dict_decal[str(ele).replace("T","P")][1]
                df["decale_"+str(ele)] = df[ele].shift(tau).ffill() 
            
            
            else:
                print("")
   
    return df   

def plotter (df,name_col1,name_col2):
    
    #Standardiser les grandeurs à afficher   
    plt.plot(df.index,df[str(name_col1)] )
    plt.plot(df.index,df[str(name_col2)] )

def borneur_df(df):
    
    #delimite la zone d'intéret du dataframe et retourne l'intervalle de celle-ci sous la forme d'une limite gauche et droite(low et high index)
    #utilise la "dérivée" de la pression pour ce faire

    
    
    new_df_std = (df-df.mean())/(df.std())
    new_df_std = new_df_std.rolling(window=20, win_type='gaussian', center=True).mean(std=5)
    derivee_pression= new_df_std["Pression"].diff(10)
    
    #fonction de scipy signal qui permet de trouver les pics. Ici, on trouve les pics correspondant aux paliers de montées et descentes en pression successives
    #On se base sur la descente en pression de fin d'etalonnage pour mesurer déterminer la limite droite
    # on se base sur le premier pic de montée en pression pour déterminer la limoite gauche
    peaker_pressure = sp.signal.find_peaks(-(derivee_pression),prominence=0.01,distance = 600)
    

    #plt.plot(new_df.index,new_df['Pression'] )
    #plt.plot(new_df_std.index,(new_df_std['diff_Pression']) )


    limit_left = peaker_pressure[0][0]
    limit_right = peaker_pressure[0][-1]-50
    
    return (limit_left,limit_right)

def montee_palier_remover(df):
    
    
    #en effectuant la dérivée on retrouve les montées de palier qui se matérialise par des pics
    new_df_std = (df-df.mean())/(df.std())
    new_df_std = new_df_std.rolling(window=20, win_type='gaussian', center=True).mean(std=5)
    derivee_pression= new_df_std["Pression"].diff(10)
    
    #le peak finder de scipy va travailler uniquement sur les pics positifs
    peaker_pressure = sp.signal.find_peaks(+(derivee_pression),prominence=0.1,distance = 1)
    
    
    #A partir du centre des pics, on crée une liste permettant de créer les intervalles hors pics
    interval = [0]

    for ele in peaker_pressure[0]:
    
        ele-20
        ele+20

        interval.append(ele-20)
        interval.append(ele+20)
    interval.append(interval[-1]+len(df))    
    
    #On crée une liste vide pour y mettre les df ne comprennant pas les pics
    list_ndf = [] 
    for k in range(0,len(interval),2):
        if k+1<len(interval):
            #print(k)
            list_ndf.append(df.iloc[interval[k]:interval[k+1]])
    
    #on crée le nouveau df amputé de ses pics:
    df_sliced = pd.concat(list_ndf)
    
    return df_sliced

def correlator_warning(df):
    a_list=[]
    for ele in df.columns:
            if ele.startswith("lambdaP") or ele.startswith("lambdaT"):
                a_list.append((ele, correlator_2(df,"Pression",str(ele))))
        
    return (a_list)

def Ndf_courbes_decalees(df):
    list_garde = []
    
    limit_left = borneur_df(df)[0]
    limit_right = borneur_df(df)[1]
    #print(limit_left,limit_right)
    
    
    df = df[limit_left:limit_right]
    df = montee_palier_remover(df)
    df = df.reset_index().drop(columns='index')
    
    
    
    for ele in df.columns:
        if ele.startswith('decale_'):
            list_garde.append(str(ele))
            
    mini_df = df[list_garde]
    for ele in mini_df.columns:
        if ele.startswith('decale_'):
            mini_df.rename(columns={str(ele):str(ele).replace("decale_","")}, inplace=True)
    mini_df['Pression'] = df['Pression']
    mini_df['Temperature'] = df['Temperature']     
    
    mini_df = mini_df[mini_df['Pression']>=0]
    mini_df = mini_df[mini_df['Pression']<=15.5]
    mini_df = mini_df.reset_index().drop(columns='index')
    return mini_df



def remove_all_above(df):
    right_stop =[]
    df=df.copy()
    new_df_std = (df-df.mean())/(df.std())
    new_df_std = new_df_std.rolling(window=20, win_type='gaussian', center=True).mean(std=5)
    
    for ele in new_df_std.columns:
        
        if ele.startswith("lambdaP") or ele.startswith("lambdaT"):
            
            

            derivee_pression= new_df_std[ele].diff(10)
    
            #fonction de scipy signal qui permet de trouver les pics. Ici, on trouve les pics correspondant aux paliers de montées et descentes en pression successives
            #On se base sur la descente en pression de fin d'etalonnage pour mesurer déterminer la limite droite
            # on se base sur le premier pic de montée en pression pour déterminer la limoite gauche
            peaker_pressure = sp.signal.find_peaks(-(derivee_pression),height=1)

            #plt.plot(new_df_std.index,(derivee_pression) )
            if len(peaker_pressure[0]) >0:
                limit_right = peaker_pressure[0][-1]
            else:
                limit_right = len(df)
            right_stop.append((str(ele),limit_right))
            
            df[str(ele)][limit_right-30:] = np.nan
    return df,right_stop
"""Modele de regression"""


from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.metrics import r2_score
import sklearn.metrics as metrics

def regression_results2(y_true, y_pred):

    # Regression metrics
    mse=metrics.mean_squared_error(y_true, y_pred,squared=True) 
    rmse=metrics.mean_squared_error(y_true, y_pred,squared=False) 
    rmse = round(rmse,4)
    r2=metrics.r2_score(y_true, y_pred)
    #print('r2: ', round(r2,4))
    #print('MSE: ', round(mse,4))
    print('RMSE: ', rmse)
    return rmse
   



def model_fit(df,capteur_id,deg_poly):
    
    df_capt=df.copy()[['lambdaP'+str(capteur_id),'lambdaT'+str(capteur_id),'Pression']]
    df_capt = df_capt.dropna()
    X_train, X_test, y_train, y_test = train_test_split(df_capt[['lambdaP'+str(capteur_id),'lambdaT'+str(capteur_id)]], df_capt['Pression'], test_size=0.2,random_state = 42)
    poly = PolynomialFeatures(degree = deg_poly) 
    X_poly = poly.fit_transform(X_train) 

    poly.fit(X_poly, y_train) 
    lin2 = linear_model.LinearRegression() 
    lin2.fit(X_poly, y_train)
    # Predicting a new result with Polynomial Regression 
    y_pred = lin2.predict(poly.fit_transform(X_test))

    #print(lin2.coef_)
    #print(len(lin2.coef_))
    rmse = regression_results2(y_test, y_pred)
    
    lin2.intercept_
    return rmse, lin2.coef_, lin2.intercept_,poly,lin2

#pd_rmse = pd.DataFrame(list_rmse,columns=['capteur','rmse'])