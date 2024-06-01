# -*- coding: utf-8 -*-
# interface
import streamlit as st
import requests
from stream_module import PlotScore, DataClients

#try:
#    from stream_module import PlotScore, DataClients
#except ModuleNotFoundError:
#    from stream_mod.stream_module import PlotScore, DataClients

# calculs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# model interpretation
import shap
from PIL import Image

in_local = False
if in_local == True:
    DataClients.ChangeDir()

# données de base
data_app_train = DataClients.LoadApplicationData(local=in_local)
id_min = data_app_train.SK_ID_CURR.min
id_max = data_app_train.SK_ID_CURR.max()

# description des données
desc_data = DataClients.LoadDescriptionData(local=in_local)

# données transformées pour l'entraînement
data = DataClients.LoadData2(local=in_local)

# shape global déjà calculées
global_shap_df = pd.read_csv(open('stream_mod/global_shap.csv', 'rb'), encoding="ISO-8859-1") 
global_relative_shap_df = pd.read_csv(open('stream_mod/global_relative_shap.csv', 'rb'), encoding="ISO-8859-1") 

# Liste de caractéristiques prédéfinies
characteristics = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'CODE_GENDER',
                   'CLIENT_AGE', 'CREDIT_TERM', 'NAME_EDUCATION_TYPE',
                   'DAYS_EMPLOYED_PERCENT', 'ANNUITY_INCOME_PERCENT']

# Interface Streamlit
st.title('Analyse de Scoring Client')

# Onglets
tab1, tab2, tab3 = st.tabs(["Comment m'utiliser ?", "Etape 1 - Choisir un client", "Etape 5 - Prédire le score d'un client"])

# mode d'emploi
with tab1:
    #with st.form(key="Introduction"):
    st.header("Utilisation de l'outil")
    texte_intro_1 = """
            Bienvenue dans l'API de scoring client ! \n
            Cette outil vous permet de choisir un client dans un échantillon 
            de la base de données 'Home Credit Data'' disponible avec le lien via ce lien :
                    """
    couleur_intro_1 = 'skyblue'#'beige'
    st.markdown(f'''
                <div style="text-align: left;">
                    <h3 style="color: {couleur_intro_1}; font-size: 60px; font-weight: bold;">
                        {texte_intro_1}
                    ''', unsafe_allow_html=True)
    
    lien = "https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction/notebook"
    texte_lien = "Lien vers les données d'origine"
    st.markdown(f"[{texte_lien}]({lien})")
    
    st.write("Pour utiliser cet outil, procédez de la manière suivante :")
    texte_intro_2 = """
            \n - Etape 1 : rendez-vous dans l'onglet 'Etape 1 - Choisir un client' pour choisi une méthode de saisi d'identifiant client
            \n - Etape 2 : dans cet onglet, rendez-vous dans l'encadré de 'Etape 2' pour choisi une méthode de saisi d'identifiant client
            \n - Etape 3 : rendez-vous dans l'encadré de 'Etape 3' pour renseigner un numéro d'identifiant client.
            \n - Etape 4 : rendez-vous dans l'encadré de 'Etape 4 (optionnel)' pour choisir une ou plusieurs caractéristiques du client choisi à l'étape 3. 
            Une fois choisi, vous pouvez afficher ces caractéristques sous forme de graphique en cliquant sur le bouton en bas de l'encadré.
            \n - Etape 5 : rendez-vous dans l'onglet 'Etape 5 - Prédire le score d'un client' pour lire le résultat de la prédiction du modèle pour ce client.
            Le résultat final s'affichera en gras comme "Accepté" si la conclusion du modèle est que le client n'aura pas de défaut de paiement de sont prêt.
            Sinon le résultat "Refusé" s'affichera.
                    """
    st.markdown(f'''
                <div style="text-align: left;">
                    <h3 style="color: {couleur_intro_1}; font-size: 50px; font-weight: bold;">
                        {texte_intro_2}
                    ''', unsafe_allow_html=True)
    
    #confirmation_bouton = st.form_submit_button(label="J'ai compris")
    
    st.header("Raccourcis clavier")
    st.write("Toutes les fonctionnalités sont accessibles par raccourci clavier :")
    texte_raccourci_1 = """ 
                \n- r : relancer l'application
                \n- Tabulation : passer à la fonctionnalité/objet suivant(e)
                \n- Sift + Tabulation : revenir à la fonctionnalité/objet précédent(e)
                \n- Entrée : clicker sur l'objet sur lequel on est positionné
                \n- Flèche droite | Flèche gauche : 
                \n    > si on est positionné sur un onglet : naviguer d'un onglet à l'autre
                \n    > si on est dans l'Etape 1 : switch la méthode de saisi id client
                \n    > si on est dans l'Etape 3 avec saisie manuelle : déplace le curseur à droite|gauche
                \n- Flèche haut | bas : 
                \n    > si on est dans l'Etape 1 : switch la méthode de saisi id client
                \n    > si on est dans l'Etape 3 avec saisie dans une liste : sélectionne un id client
                \n    > n'importe où ailleur dans la page : scroll vers le haut/le bas
                \n- Ctrl + (bouton +/-) : zoom/dé-zoom
                    """
    st.markdown(f'''
                <div style="text-align: left;">
                    <h3 style="color: {couleur_intro_1}; font-size: 12px; font-weight: bold;">
                        {texte_raccourci_1}
                    ''', unsafe_allow_html=True)


with tab2:
    st.header('Choisir le client à analyser')
    
    with st.form(key="Option de sélection"):
        st.header('Etape 2')
        # Option de sélection
        selected_option = st.radio(
            "Choisissez une méthode pour entrer l'identifiant client:",
            ("Sélectionner dans une liste d'identifiants", "Entrer manuellement un identifiant")
        )
        option_bouton = st.form_submit_button(label="Confirmer la méthode de saisie d'identifiant")
        
    # Saisie du numéro d'identification client
    with st.form(key="Sélection d'un client"):
        st.header('Etape 3')
        st.write(f'Veuillez {selected_option} un identifiant client.')
        
        texte_etape_3 = f'Veuillez {selected_option} un identifiant client.'
        st.markdown(f'''
                    <div style="text-align: left;">
                        <h3 style="color: {couleur_intro_1}; font-size: 16px">
                            {texte_etape_3}
                        ''', unsafe_allow_html=True)
        
        client_ids = data_app_train.SK_ID_CURR.tolist()
        # Si l'utilisateur choisit de sélectionner dans la liste
        if selected_option == "Sélectionner dans une liste d'identifiants":
            client_id = st.selectbox("Sélectionnez l'identifiant client:", client_ids)
        
        # Si l'utilisateur choisit de saisir manuellement
        elif selected_option == "Entrer manuellement un identifiant":
            client_id = st.number_input("Entrez l'identifiant client:", min_value=id_min)
        
        confirm_button = st.form_submit_button(label="Confirmer le choix d'identifiant")
        
    with st.form(key="Caractéristiques client"):
        st.header('Etape 4')
        if client_id in data_app_train['SK_ID_CURR'].values:
            # Données du client sélectionné
            client_data = data_app_train[data_app_train['SK_ID_CURR'] == client_id].iloc[0]
            client_data_for_prediction = data[data['SK_ID_CURR'] == client_id].drop('SK_ID_CURR', axis=1).iloc[0]
            
            st.subheader('Sélectionnez des caractéristiques à afficher')
            
            texte_etape_4 = """ 
                        Grâce à la zone de sélection ci-dessous, vous pouvez ajouter ajouter des caractéristiques à\n
                        afficher via le menu déroulant disponible en cliquant sur la flèche sur la droite.\n
                        Vous pouvez également retirer des caractéristques pour qu'elles ne soient plus affichées.\n
                        Pour cela, cliquez sur la croix à droite de l'étiquette d'une caractéristique.
                            """
            st.markdown(f'''
                        <div style="text-align: left;">
                            <h3 style="color: {couleur_intro_1}; font-size: 12px; font-weight: bold;">
                                {texte_etape_4}
                            ''', unsafe_allow_html=True)
            
            list_caracteristiques = data_app_train.columns.tolist()
            list_caracteristiques.sort()
            selection = st.multiselect('Ajouter ou retirer une caractéristique :', list_caracteristiques, default=characteristics[:-1])
            
             
        else:
            color = 'orange'
            st.markdown(f'''
                        <div style="text-align: left;">
                            <h3 style="color: {color}; font-size: 24px; font-weight: bold;">
                                {"Oups ! ce client n'existe pas dans la base de données... Veuillez en choisir un autre :)"}
                            </h3>
                        </div>
                        ''', unsafe_allow_html=True)
                                
        texte_etape_4 = "En cliquant sur le bouton ci-dessous, les caractéristiques du client s'afficheront en dessous."
        st.markdown(f'''
                    <div style="text-align: left;">
                        <h3 style="color: {couleur_intro_1}; font-size: 16px">
                            {texte_etape_4}
                        ''', unsafe_allow_html=True)
        
        select_button = st.form_submit_button(label='Afficher caractéristiques choisies')
    
    if select_button:
        plt.close('all')
        try:
            for selected_characteristic in selection:            
                
                mask_1 = desc_data.Table == 'application_{train|test}.csv'
                mask_2 = desc_data.Row == selected_characteristic
                desc = desc_data.loc[mask_1 & mask_2, 'Description'].values[0]
                desc_title = f'Caractéristique {selected_characteristic} :'
                #desc_content = f"Description d'origine : {desc}"
                desc_content = f'<span style="text-decoration: underline;">Description d\'origine </span>: {desc}'
                
                st.markdown(f'''
                            <div style="text-align: left;">
                                <h3 style="color: {'white'}; font-size: 18px">
                                    {desc_title}
                                </h3>
                                <h3 style="color: {couleur_intro_1}; font-size: 16px">
                                    {desc_content}
                                </h3>
                                ''', unsafe_allow_html=True)
                
                # Affichage de l'histogramme
                fig, ax = plt.subplots()
                if pd.api.types.is_numeric_dtype(data_app_train[selected_characteristic]):
                    # écriture d'une descrption pour interpretation graphique
                    # valeur :
                    client_value = client_data[selected_characteristic]
                    mean_value = data_app_train[selected_characteristic].mean()
                    comparator = 'inférieure'
                    if client_value > mean_value: comparator = 'supérieure'
                    desc_interprete = f"""<span style="text-decoration: underline;">Lecture graphique</span> : le client à une valeur de 
                                        {selected_characteristic} de {round(client_value,2)} qui est {comparator}
                                        à la moyenne de {round(mean_value,2)} (tout clients confondus, soit un total de {data_app_train.shape[0]} clients)."""
                                        
                    st.markdown(f'''
                                <div style="text-align: left;">
                                    <h3 style="color: {couleur_intro_1}; font-size: 16px">
                                        {desc_interprete}
                                    </h3>
                                    ''', unsafe_allow_html=True)
                                    
                    # Histogramme pour les caractéristiques numériques
                    sns.histplot(data_app_train[selected_characteristic], kde=False, ax=ax)
                    client_value_line = ax.axvline(client_value, color='r', linestyle='--', label=f'Client ({round(client_value, 2)})')
                    client_mean_line = ax.axvline(mean_value, color='black', linestyle='-', label=f'Moyenne ({round(mean_value, 2)})')
                    ax.legend(handles=[client_value_line, client_mean_line])
                    ax.set_title(f'Histogramme de {selected_characteristic}')
                else:
                    # écriture d'une descrption pour interpretation graphique
                    # valeur :
                    client_value = client_data[selected_characteristic]
                    other_values = data_app_train[selected_characteristic].unique()
                    
                    val_count = data_app_train[selected_characteristic].value_counts()
                    majority_class = val_count.loc[val_count == val_count.max()].index[0]
                    majority_value = val_count.loc[val_count == val_count.max()].values[0]
                    minority_class = val_count.loc[val_count == val_count.min()].index[0]
                    minority_value = val_count.loc[val_count == val_count.min()].values[0]

                    desc_interprete = f"""<span style="text-decoration: underline;">Lecture graphique</span> : le client à une valeur de 
                                        {selected_characteristic} de {client_value}. Les autres valeurs possibles sont {other_values}.
                                        La classe majoritaire est {majority_class} comptant {majority_value} clients.
                                        La classe minoritaire est {minority_class} comptant {minority_value} clients.
                                        Au total les données comptent {data_app_train.shape[0]} clients."""
                                        
                    st.markdown(f'''
                                <div style="text-align: left;">
                                    <h3 style="color: {couleur_intro_1}; font-size: 16px">
                                        {desc_interprete}
                                    </h3>
                                    ''', unsafe_allow_html=True)
                    # Histogramme pour les caractéristiques de type objet
                    sns.countplot(data_app_train[selected_characteristic], ax=ax)
                    client_value_line = ax.axhline(client_data[selected_characteristic], color='r', linestyle='--', label='valeur client')
                    ax.legend(handles=[client_value_line])
                    ax.set_title(f'Fréquence des catégories pour {selected_characteristic}')
                
                st.pyplot(fig)
                
        except NameError:
            st.write("Aucune caractéristique à afficher car le client n'est pas enregistré.")                                                                                                                              

    # Créer un scatterplot avec Seaborn                   
    with st.form(key="Analyse bi-variée"):
        st.subheader("Analyse entre deux caractéristiques")
        
        text_bivar = """Ci-dessous, vous pouvez choisir deux caractéristiques parmi celles ayant la plus gr"""
        st.markdown(f'''
                    <div style="text-align: left;">
                        <h3 style="color: {couleur_intro_1}; font-size: 16px">
                            {text_bivar}
                        
                        ''', unsafe_allow_html=True)
        
        feature1 = st.selectbox("Sélectionnez une première caractéristique:", selection)
        feature2 = st.selectbox("Sélectionnez une seconde caractéristique:", selection)
        color = st.selectbox("Sélectionnez une troisième caractéristique pour graduer les couleurs:", list_caracteristiques)
        
        select_button = st.form_submit_button(label='Confronter ces caractéristiques')
        
        if select_button:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data_app_train[feature1], y=data_app_train[feature2], hue=data_app_train[color], alpha=0.6)
            plt.title(f'{feature1} vs. {feature2}')
            plt.xlabel(f'{feature1}')
            plt.ylabel(f'{feature2}')
            # Afficher le scatterplot dans Streamlit
            st.pyplot(plt)

with tab3:
    def convert_to_native_type(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return 0
        return value
    
    if client_id in data_app_train['SK_ID_CURR'].values:
        
        features = []
        for val in client_data_for_prediction.values:
            features.append(val)
        
        st.header('Résultat de l\'analyse')
    
        if client_id in data_app_train['SK_ID_CURR'].values:
            # Affichage de l'identifiant du client
            st.subheader(f"ID du client: {client_id}")
    
            # Prédiction du score de défaut de paiement
            features = [convert_to_native_type(item) for item in features]
            client_features = {'features': {str(key): value for key, value in zip(client_data_for_prediction.index, features)}} #{"features": features}
            
            url = 'https://apigamba-6f486e3c76df.herokuapp.com/predict'
            if in_local == True: url = 'http://127.0.0.1:8000/predict'
            response = requests.post(url, json=client_features)
            
            if response.status_code == 200:
                
                # récupération des réponses
                api_response = response.json()
                prediction = api_response['prediction']
                probabilite = api_response['probability']
                shap_values =  np.array(api_response['shap_values'][0])
                
                # jauge score 
                st.subheader(f"Score de défaut de paiement: {probabilite:.2f}")
                st.plotly_chart(PlotScore.jauge_bar(probabilite))
        
                level = 'inférieur'
                status = 'Accepté'
                r = 'green'
                if prediction == 1: 
                    level = 'supérieur'
                    status = 'Refusé'
                    r = 'red'
                #st.subheader(f"{status}", color = r)
                st.markdown(f'''
                            <div style="text-align: center;">
                                <h3 style="color: {r}; font-size: 60px; font-weight: bold;">
                                    {status}
                                </h3>
                            </div>
                            ''', unsafe_allow_html=True)
                #st.markdown(f'<h3 style="color: {r};">{status}</h3>', unsafe_allow_html=True)
                texte_jauge = f"Le client n° {client_id} a une probabilité de défaut de paiement de {probabilite:.2f} {level} au seuil de 0,411. Status crédit : {status}."
                st.markdown(f'''
                            <div style="text-align: left;">
                                <h3 style="color: {couleur_intro_1}; font-size: 16px">
                                    {texte_jauge}
                                ''', unsafe_allow_html=True)
                
                # Affichage du waterfall plot des valeurs SHAP
                st.subheader("Interprêtation du résultat (SHAP values)")
                
                mean_log_odd = -0.905
                mean_proba = round(1 / (1 + np.exp(-mean_log_odd)), 2)
                
                text_local_shap_1 = f"""<span style="text-decoration: underline;">Contexte de l'interprêtation graphique</span> : le graphique ci-dessous
                                montre la contribution relative de chaque caractéristique dans le score de probabilité affiché en haut
                                de cette page. Selon la valeur d'une caractéristique, sa contribution dans la valeure de sortie finale
                                ne sera pas nécessairement la même d'un individu à un autre.
                                La valeur de contribution d'une caractéristique est relative à la valeur de sortie moyenne du modèle
                                servant de référence. Cette valeur de référence est de {mean_log_odd} soit une probabilité moyenne de 
                                {mean_proba}. Les caractéristiques utilisées dans ce graphique sont issues d'une série de transformations
                                pour augmenter les performances du modèle. Ainsi, les noms des variables peuvent légèrement
                                différer des noms d'origine.
                                Ci-dessous vous pouvez choisir le nombre de caractéristiques à faire afficher pour l'interprêtation du 
                                résultat."""
                                    
                st.markdown(f'''
                            <div style="text-align: left;">
                                <h3 style="color: {couleur_intro_1}; font-size: 16px">
                                    {text_local_shap_1}
                                </h3>
                                ''', unsafe_allow_html=True)
                
                max_diplay = st.number_input("nombre de caractéristiques à afficher:", min_value=5)
                
                # Création d'un Waterfall plot avec SHAP
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.waterfall_plot(shap.Explanation(values=shap_values, 
                                                     base_values=-0.9053249661763899, #np.mean(shap_values), 
                                                     data=features, 
                                                     feature_names=client_data_for_prediction.index), 
                                    max_display=max_diplay)
                ax.set_title(f'Caractéristiques principales pour prédire le client n°{client_id}')
                st.pyplot(fig)
                
                # description textuelle du waterfall
                info_dict = {'name' : client_data_for_prediction.index.tolist(),
                             'data_value' : features,
                             'shap_value' : shap_values,
                             'shap_value_abs' : np.abs(shap_values)}
                tab_synthese = pd.DataFrame(info_dict)
                tab_synthese = tab_synthese.sort_values('shap_value_abs', ascending=False).head(max_diplay)
                tab_synthese = tab_synthese.reset_index(drop=True)
                
                text_local_shap_2 = """<span style="text-decoration: underline;">Lecture graphique</span> : dans l'ordre décroissant
                                    de la contribution absolue, on a : \n"""
                st.markdown(f'''
                            <div style="text-align: left;">
                                <h3 style="color: {couleur_intro_1}; font-size: 16px">
                                    {text_local_shap_2}
                                
                                ''', unsafe_allow_html=True)
                                
                for i in range(max_diplay-1):
                    
                    feature_name = tab_synthese.loc[tab_synthese.index[i], 'name']
                    feature_val = round(tab_synthese.loc[tab_synthese.index[i], 'data_value'], 2)
                    shap_val = round(tab_synthese.loc[tab_synthese.index[i], 'shap_value'], 2)
                    signe = 'positivement'
                    if shap_val < 0: signe = 'negativement'
                    text_local_shap_i = f"""
                    - {feature_name} d'une valeur de {feature_val} contribut {signe} de {shap_val} points.
                    """
                    
                    if shap_val < 0:    
                        st.markdown(f'''
                                    <div style="text-align: left;">
                                        <h3 style="color: {'deepskyblue'}; font-size: 16px">
                                            {text_local_shap_i}
                                        
                                        ''', unsafe_allow_html=True) # lightgreen  limegreen
                    else:    
                        st.markdown(f'''
                                    <div style="text-align: left;">
                                        <h3 style="color: {'hotpink'}; font-size: 16px">
                                            {text_local_shap_i}
                                        
                                        ''', unsafe_allow_html=True)
                           
                
                
                st.subheader("Contributions générales des caractéristiques")
                
                # image de l'importance globale
                image = Image.open('stream_mod/GlobalShap.png')
                st.image(image, caption='Importances globales absolues des caractéristiques', use_column_width=False, 
                         output_format="PNG", width=900)
                
                # beeswarm plot
                # image de l'importance globale
                image = Image.open('stream_mod/GlobalShap_beeswarm.png')
                st.image(image, caption='Importances globales relatives des caractéristiques', use_column_width=False, 
                         output_format="PNG", width=900)
                
                # description de l'importance globale
                
                text_global_shap_1 = """<span style="text-decoration: underline;">Explication</span> : la première figure nommée
                'Importances globales absolues des caractéristiques' montre la contribution moyenne absolue de chaque caractéristique 
                dans l'ordre décroissant. La seconde figure montre la contribution relative de ces caractéristiques, graduée en couleur
                 selon leur valeur d'origine. Voici dans l'ordre décroissante la lecture des deux graphiques pour chaque caractéristique :\n"""
                
                st.markdown(f'''
                            <div style="text-align: left;">
                                <h3 style="color: {couleur_intro_1}; font-size: 16px">
                                    {text_global_shap_1}
                                
                                ''', unsafe_allow_html=True)
                                
                for i in range(global_shap_df.shape[0]):
                    
                    feature_name = global_shap_df.loc[global_shap_df.index[i], 'index']
                    feature_val_abs = global_shap_df.loc[global_shap_df.index[i], 'absolute_importance_mean']
                    feature_val_relatif = global_relative_shap_df.loc[global_relative_shap_df.index[i], 'relative_importance_mean']

                    signe = 'augmente'
                    if feature_val_relatif < 0: signe = 'diminue'
                    text_global_shap_i = f"""
                    - {i+1}) {feature_name} a une contribution absolue de {round(feature_val_abs, 2)}. \n
                    Le plus souvent {feature_name} {signe} la prédiction du modèle.
                    """
                    
                    if feature_val_relatif < 0:    
                        st.markdown(f'''
                                    <div style="text-align: left;">
                                        <h3 style="color: {'deepskyblue'}; font-size: 16px">
                                            {text_global_shap_i}
                                        
                                        ''', unsafe_allow_html=True) # lightgreen  limegreen
                    else:    
                        st.markdown(f'''
                                    <div style="text-align: left;">
                                        <h3 style="color: {'hotpink'}; font-size: 16px">
                                            {text_global_shap_i}
                                        
                                        ''', unsafe_allow_html=True)
                
            else:
                st.write("Erreur lors de la prédiction.")
                st.write(response.json())
