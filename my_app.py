#script streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xlrd
#les fonctions dévéloppées pour le traitement des données sont dans streamlit_functions.py
import streamlit_functions as sf
#%matplotlib inline

import scipy as sp


from sklearn import preprocessing
pd.options.mode.chained_assignment = None
from pathlib import Path
import os
import re
import itertools
import base64
from bokeh.models import LinearAxis, Range1d
from bokeh.layouts import gridplot
from bokeh.layouts import row
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Dark2_5 as palette
from bokeh.palettes import Magma256
import plotly.graph_objects as go


st.title(" Test App Web Etalonnage Capteur CFOP")


uploaded_file  = st.file_uploader("Déposez les fichiers sous la forme Pression_30.xls (en provenance de la sonde Additel) ,Channel4_30.txt(en provenance du BraggLogger)", type=["xls", "txt"], accept_multiple_files=True)
st.write(":inbox_tray: Les fichiers en provenance du bragglogger peuvent s'intituler ChannelX_TTT avec\n X un chiffre entre 0 et 9 et TTT, la temperature en °C")
st.write(":inbox_tray: Les fichiers Pression et Channel doivent se terminer par une température pour être associés entre eux. ")


def load_fichiers(uploaded_file):

    fichiers_charges=[]
    if len(uploaded_file)==0:
        st.write(":no_entry_sign: Aucun fichier chargé")
        exemple = st.checkbox("Cochez cette case pour charger des fichiers à titre d'exemple")
        if exemple==True:
            data_folder= Path("Data./")
            fichier_txt = list(data_folder.glob('**/*.txt'))
            fichier_xls = list(data_folder.glob('**/*.xls'))
    
    
            file_tot = fichier_txt+fichier_xls
            fichiers_charges=file_tot
              
        
    elif len(uploaded_file)==1 : 
        st.write("Au moins 1 fichier manquant")
        
    
    else: 
        fichiers_charges=uploaded_file
        
    return fichiers_charges

list_fichiers = load_fichiers(uploaded_file)
if list_fichiers==[]:
    st.stop()




@st.cache
def trie_fichiers(list_fichiers):
    list_fichiers_name=[]

    for ele in list_fichiers:
        list_fichiers_name.append(ele.name)
    list_couple = sf.trait_list_file(list_fichiers)
    list_traitee_flatted=[]
    list_traitee=[]
    for line in list_couple:
        list_traitee.append((line[0].name,line[1].name))
        for ele in line:
            list_traitee_flatted.append(ele.name)


    
    fichiers_non_pris_en_compte = list(set(list_fichiers_name) - set(list_traitee_flatted))
    df_list_traitee = pd.DataFrame(list_traitee, columns=["Channel","Pression"])
    return fichiers_non_pris_en_compte,df_list_traitee

fichiers_non_pris_en_compte,df_list_traitee = trie_fichiers(list_fichiers)



for ele in fichiers_non_pris_en_compte:
    st.write("fichier ignoré : "+str(ele))

st.text("------------------------------------------------------------------------------------------")
st.text("Couples Channel-Pression considérés pour le calcul; une ligne correspondant à un couple")
st.dataframe(df_list_traitee)
st.text("------------------------------------------------------------------------------------------")

@st.cache
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">lien_tableau</a>'
    return href

@st.cache(allow_output_mutation=True)
#operation de mutation temperature pour affichage en categorie de temp dans plotlysation
def load_data(list_fichiers):
    df_final_NT,df_final,dict_df = sf.general_concat (list_fichiers)
    return df_final_NT,df_final,dict_df

#On recupère les df non traités et traités ainsi que le dictionnaire donnant les coefficien
df_final_NT,df_final,dict_df = load_data(list_fichiers)


st.subheader("Tableau donnant le coefficient de corrélation entre lambdaP, T  et la Pression pour différentes températures")

st.write(dict_df)
st.markdown(get_table_download_link(dict_df), unsafe_allow_html=True)

#u=on recupère les températures pérésentes dans le dataframe
@st.cache
def list_temp(df):
    list_temp_df=list(df["Temperature"].unique())
    list_temp_df.sort()
    return list_temp_df
list_temp_df = list_temp(df_final)

@st.cache
def capteur_id_nbr(df):

    plop=[]
    for ele in list(df_final.columns):
        if ele.startswith("lambdaP"):
            plop.append(ele)
    nbr_capt_id = [i for i in range(0,len(plop),1)]
    return nbr_capt_id

#recup list numero de capteur
nbr_capt_id = capteur_id_nbr(df_final)

def verif_recalage(df):
    y_overlimit = 0.0
    df=df.copy()
    list_graph_capt=[]
    for num_capt in nbr_capt_id:
        list_graph_temp=[]
        for k,temp in enumerate(list_temp_df):
            
            df_temp = df[df["Temperature"]==temp]
            #df_aff= sf.standardize_df_MinMax(df_temp)
            df_aff = df_temp
            df_aff =df_aff.reset_index(drop=True)
            #st.write(df_aff)
            s = figure(width=250, plot_height=250, title="Temperature "+str(temp), x_axis_label='temps(UA)', y_axis_label="capteur "+str(num_capt))
            if k==0:
                s.line(df_aff.index, df_aff["lambdaP"+str(num_capt)], legend_label="lambdaP"+str(num_capt),line_color="blue")
                s.line(df_aff.index, df_aff["lambdaT"+str(num_capt)], legend_label="lambdaT"+str(num_capt),line_color="red")
                s.y_range = Range1d(df_aff["lambdaP"+str(num_capt)].min() * (1 - y_overlimit), df_aff["lambdaT"+str(num_capt)].max() * (1 + y_overlimit))
                
                
                s.extra_y_ranges = {"Pression": Range1d(start=df_aff["Pression"].min() * (1 - y_overlimit),end=df_aff["Pression"].max() * (1 + y_overlimit))}
                s.add_layout(LinearAxis(y_range_name = "Pression"), "right")
                s.line(df_aff.index, df_aff["Pression"], legend_label="Pression",y_range_name="Pression",line_color="black")
                
            else:
                s.line(df_aff.index, df_aff["lambdaP"+str(num_capt)],line_color="blue")
                s.line(df_aff.index, df_aff["lambdaT"+str(num_capt)],line_color="red")
                s.y_range = Range1d(df_aff["lambdaP"+str(num_capt)].min() * (1 - y_overlimit), df_aff["lambdaT"+str(num_capt)].max() * (1 + y_overlimit))


                s.extra_y_ranges = {"Pression": Range1d(start=df_aff["Pression"].min() * (1 - y_overlimit),end=df_aff["Pression"].max() * (1 + y_overlimit))}
                s.add_layout(LinearAxis(y_range_name = "Pression"), "right")
                s.line(df_aff.index, df_aff["Pression"],y_range_name="Pression",line_color="black")
            list_graph_temp.append(s)
        list_graph_capt.append(list_graph_temp)
    
    layout = gridplot(list_graph_capt,toolbar_location ='right')
    

    return layout


checker0 = st.checkbox('Cochez pour observer la variation de lambdaP,T et de la Pression en fonction du temps',value=False)
if checker0==True:
    
    
    fig= verif_recalage(df_final)
    st.bokeh_chart(fig,use_container_width=False)

def courbes_capteurs_pression(df):
    df= df

    temp_list =  list_temp_df
    
    

    features = [z for z in list(df.columns) if str(z)!="Pression" and str(z)!="Temperature"]
    list_fig=[]

    for name_col in features:

            fig =plt.figure(figsize=(70,5),dpi=450)


            for  i,temp in enumerate(temp_list) :


                df_display=df[df["Temperature"]==temp]
                df_display=sf.standardize_df_MinMax(df_display)
                df_display = df_display.reset_index(drop=True)





                ax = fig.add_subplot(1,len(features),i+1)
                fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.50, hspace=None)

                x=df_display.index
                y1=df_display["Pression"]
                ax.plot(x,y1,"blue",label="Pression")


                y2=df_display[str(name_col)]
                ax.plot(x,y2,"red",label=str(name_col))



                leg = ax.legend()
                ax.set_title("Temperature "+str(int(temp))+"C°")
                #ax.set_xlim(0, 50)
                #ax.set_ylim(0, 30)
                ax.set_xlabel("temps (UA)")
                ax.set_ylabel("UA (normalisé)")
            list_fig.append(fig)
    return list_fig

#for ele in courbes_capteurs_pression(df_final):
    #st.pyplot(ele)



st.text("------------------------------------------------------------------------------------------")
st.subheader("Tableau final regroupant les lambdaP,T recalés avec la mesure de Pression")
st.write(df_final)

@st.cache
def columns_checker(df,dict_df):
    listos1 = list(df.columns)
    listos1.remove("Temperature")
    listos1.remove("Pression")
    listos2 = list(dict_df.index)
    non_pris = set(listos2) - set(listos1) 
    if non_pris is not None:
        rep,n = ("Les lambdasP,T suivants ne figurent pas dans le tableau final car le coefficient de corrélation\n avec la Pression est trop faible indiquant potentiellement un defaut du capteur :" ,non_pris)
    else:
        rep,n = ("Terminé","")
    return str(rep)+str(n)

st.write(columns_checker(df_final,dict_df))

st.markdown(get_table_download_link(df_final), unsafe_allow_html=True)
st.text("------------------------------------------------------------------------------------------")

@st.cache(allow_output_mutation=True)
def bokehtisation(df):
    dict_fig_boke={}
    for ele in df.columns:
        p = figure(title=str(ele)+" en fonction de la Pression pour diverses temperatures", x_axis_label='Pression', y_axis_label=str(ele))
        
        colors = itertools.cycle(palette)  
        
        for temp, color in zip(list_temp_df,colors):   
            p.scatter(df[df["Temperature"]==temp]["Pression"], df[df["Temperature"]==temp][ele], legend_label=str(ele)+" Temp={}".format(temp),fill_color=color,line_color=color)

        dict_fig_boke[ele]=p

    return dict_fig_boke
 
dictus = bokehtisation(df_final)


checker1 = st.checkbox('Cochez pour observer la varation lambdaP,T en fonction de la pression pour différentes températures')
if checker1==True:
    column_selector1 = st.selectbox('Selectionnez la valeur traçée en fonction de la Pression', list(df_final.columns),key="column_selector1")
    #fig_boke = bokehtisation(df_final,column_selector1)


    st.bokeh_chart(dictus[column_selector1])









st.text("------------------------------------------------------------------------------------------")
import plotly
import plotly.express as px

@st.cache
def plotlysation(df,capteur_id):

    x="lambdaP"+str(capteur_id)
    y="lambdaT"+str(capteur_id)
    
    df["Temperature"]=df["Temperature"].astype(str)
    fig0 = px.scatter_3d(df,x=x,y=y,z="Pression",color="Temperature")
    df["Temperature"]=df["Temperature"].astype(float)

    fig0.update_layout(title="Pression en fonction de lambdaP,T", autosize=False,
    width=800, height=800,
    margin=dict(l=100, r=100, b=100, t=100))
    return fig0




checker2 = st.checkbox('Cochez pour observer la varation lambdaP,T graph3D')
if checker2==True:
    column_selector2 = st.selectbox('Numéro du capteur',nbr_capt_id )


    fig0 = plotlysation(df_final,column_selector2)
    st.plotly_chart(fig0)
st.text("------------------------------------------------------------------------------------------")




@st.cache(allow_output_mutation=True)
def fitteur_traceur(df,column_selector3,deg_poly_selector ):
    #on fit avec un polynome 
    num_capteur= column_selector3
    df=df.copy()

    rmse, coef, intercept,poly,lin2 = sf.model_fit(df,num_capteur,deg_poly_selector)
    

    
    
    traceur=True
    if traceur==True:

        N = 100
        
        borne_inf_P = df["lambdaP"+str(num_capteur)].min()
        borne_sup_P =df["lambdaP"+str(num_capteur)].max()
        plageP=np.abs(borne_sup_P-borne_inf_P)
        borne_inf_T = df["lambdaT"+str(num_capteur)].min()
        borne_sup_T =df["lambdaT"+str(num_capteur)].max()
        plageT=np.abs(borne_sup_T-borne_inf_T)

        predict_x0, predict_x1 = np.meshgrid(np.linspace(borne_inf_P, borne_sup_P, N), 
                                            np.linspace(borne_inf_T, borne_sup_T, N))
        predict_x = np.concatenate((predict_x0.reshape(-1, 1), 
                                    predict_x1.reshape(-1, 1)), 
                                axis=1)

        predict_x_ = poly.fit_transform(predict_x)
        
        predict_y = lin2.predict(predict_x_)
        predict_y[predict_y>16]=np.nan
        predict_y[predict_y<-1]=np.nan
        

        
        fig = go.Figure(data=[go.Surface(z= predict_y.reshape(predict_x0.shape), x=predict_x0, y=predict_x1)])
        



        
        fig2  = px.scatter_3d(df,x="lambdaP"+str(num_capteur),y="lambdaT"+str(num_capteur),z="Pression")
        
        fig.add_trace(fig2.data[0])

        fig.update_layout(title="Fit polynomiale ordre "+str(poly.degree)+" données lambdaP,T capteur "+str(num_capteur), 
                            autosize=False,
                            
                            width=800, 
                            height=800,
                            margin=dict(l=100, r=100, b=100, t=100),
                            scene=dict(
                                xaxis = dict(nticks=6, range=[borne_inf_P-(plageP/2),borne_sup_P+(plageP/2)],),
                                yaxis = dict(nticks=6, range=[borne_inf_T-(plageT/2),borne_sup_T+(plageT/2)],),
                                zaxis = dict(nticks=6, range=[-2,df["Pression"].max()+5],),

                                xaxis_title="lambdaP"+str(num_capteur),
                                yaxis_title="lambdaT"+str(num_capteur),
                                zaxis_title="Pression"
                                ),
                            
                            
                            

            
                        
                        
                        )
        
        
    return num_capteur,rmse,coef,intercept,poly,fig
@st.cache(allow_output_mutation=True)
def presentation_coeffs(num_capteur,rmse,coef,intercept,poly):
    
    coef_pd = pd.DataFrame(coef,columns=["Coefficients"])
    coef_pd["Coefficients"].iloc[0]=intercept
    coef_name=pd.DataFrame(poly.get_feature_names(["lambdaP"+str(num_capteur),"lambdaT"+str(num_capteur)]))
    coef_pd["Variable"]=coef_name
    coef_pd["NumCapt"]=num_capteur
    coef_pd["Rmse"]=rmse
    coef_pd["DegPoly"]=poly.degree

    return coef_pd

column_selector3 = st.selectbox('Numéro du capteur choisi',nbr_capt_id )
list_degres=[1,2,3,4,5]
deg_poly_selector = st.selectbox('degré du polynôme de fit',list_degres)


@st.cache(allow_output_mutation=True)
def calcul_fit_global(df,nbr_capt_id,list_degres):
    listus_df=[]

    for i in nbr_capt_id:
        for j in list_degres:
            num_capteur,rmse,coef,intercept,poly,fig = fitteur_traceur(df,i,j)
            
            a=presentation_coeffs(i,rmse,coef,intercept,poly)
            listus_df.append((a,i,j,fig))
    dfus=[]
    for ele in listus_df:
        dfus.append(ele[0])

    concatos=pd.concat(dfus)
    return concatos,listus_df

concatos,listus_df = calcul_fit_global(df_final,nbr_capt_id,list_degres)

df_display = concatos[(concatos["NumCapt"]==column_selector3)&(concatos["DegPoly"]==deg_poly_selector)]
st.dataframe(df_display)
st.markdown(get_table_download_link(df_display), unsafe_allow_html=True)
#st.dataframe(concatos[concatos["DegPoly"]==1])

def trace_fit(num_capt,degpoly,listus_df):
    for ele in listus_df:
        if ele[1]==num_capt:
            if ele[2]==degpoly:
                st.plotly_chart(ele[3])
            
        
trace_fit(column_selector3,deg_poly_selector,listus_df)

st.text("------------------------------------------------------------------------------------------")

@st.cache(allow_output_mutation=True)

def courbes_niveau(df,num_capt,deg_poly_fit):
    df=df.fillna(0)
    p1 = figure(title="Courbes de niveaux", x_axis_label="lambdaP"+str(num_capt)+" (nm)", y_axis_label="lambdaT"+str(num_capt)+" (nm)")
    
                


    
    count_col = int(256/15)-1
    for k  in range(0,16,1):
        df_display1 = df[df["Pression"].between(k,k+1.5)]
        
        if df_display1.empty is False:

            df_display1 = df_display1[df_display1["lambdaT"+str(num_capt)]>0]
            df_display1 = df_display1[df_display1["lambdaP"+str(num_capt)]>0]


            p1.scatter(df_display1["lambdaT"+str(num_capt)], df_display1["lambdaP"+str(num_capt)], color=Magma256[k*count_col],legend_label="Intervalle de pression "+"["+str(k)+":"+str(k+1.5)+"]")
            x1 = df_display1["lambdaT"+str(num_capt)].values
            y1= df_display1["lambdaP"+str(num_capt)].values
            z1 = np.polyfit(x1, y1, deg_poly_fit)
            f1 = np.poly1d(z1)
            x_fit1 = np.linspace(np.min(x1), np.max(x1), 1000)
            y_fit1 = [f1(_x) for _x in x_fit1]
            p1.line(x_fit1, y_fit1, color=Magma256[k*count_col])

   
    return  p1

checker3 = st.checkbox('Cochez pour observer les courbes de niveaux')
if checker3==True:
    st.bokeh_chart(courbes_niveau(df_final,column_selector3,1))



































