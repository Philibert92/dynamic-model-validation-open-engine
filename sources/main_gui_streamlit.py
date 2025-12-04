# Copyright (c) 2022-2025, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the dynamic-model-validation-engine project.

"""
Main function to run the GUI of the Dynawo Model validation toolbox.
"""

import os
import shutil
import zipfile
import pandas as pd
import streamlit as st
from PIL import Image
from settings import Settings
from util_functions import get_measured_data, get_col_name_p_q, sample_df, pearson_corr, similarity_metrics, rmse, \
    clean_directory, get_file_hash
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from logger_management import initialize_streamlit_logger, get_streamlit_logs, clear_streamlit_logger
from dynawo_functions import run_dynawo, run_custom_dynawo, \
    get_parameters_sets, DynawoParam, DynawoFailedException
from sensitivity_analysis import run_sensitivity_analysis
from automatic_calibration import run_parameter_calibration, OptimMethod
from st_keyup import st_keyup


app_title = "Dynawo Model Validation Tool"
st.set_page_config(page_title=app_title, layout="wide")


def create_webpage_header():
    col_1, col_2, col_3, col_4 = st.columns([14, 4, 4, 4])
    script_directory = os.path.dirname(os.path.abspath(__file__))
    with col_1:
        st.title("Model Validation Tool")
    with col_2:
        st.write("")  # for spacing
    with col_3:
        esic_logo_path = os.path.join(script_directory, "..", "resources", "wsu_logo.png")
        esic_logo = Image.open(esic_logo_path)
        st.image(esic_logo, use_container_width=False)
    with col_4:
        rte_logo_path = os.path.join(script_directory, "..", "resources", "rte_logo.png")
        rte_logo = Image.open(rte_logo_path)
        st.image(rte_logo, use_container_width=False, width=80)


def reinit_analysis(
    temp_folders_base_cases,
    temp_folders_sensitivity_cases,
    temp_folders_calibration_cases,
    temp_folders_custom_calibration_cases
):
    """
    Réinitialise l'analyse :
    - Vide les dossiers temporaires utilisés dans l'analyse
    - Supprime toutes les variables du session_state de Streamlit contenant les données, résultats et logs.
    - Nettoie les loggers spécifiques à l'application.
    """

    # Nettoyage des dossiers
    for case in temp_folders_base_cases:
        clean_directory(temp_folders_base_cases[case])
        clean_directory(temp_folders_sensitivity_cases[case])
        clean_directory(temp_folders_calibration_cases[case])
        clean_directory(temp_folders_custom_calibration_cases[case])

    # Suppression des données stockées dans session_state 
    if "measured_data_df" in st.session_state:
        del st.session_state["measured_data_df"]
    if "jobs_files" in st.session_state:
        del st.session_state["jobs_files"]
    if "iidm_file" in st.session_state:
        del st.session_state["iidm_file"]
    if "dyd_file" in st.session_state:
        del st.session_state["dyd_file"]
    if "par_file" in st.session_state:
        del st.session_state["par_file"]
    if "parameters_sets" in st.session_state:
        del st.session_state["parameters_sets"]
    if "crv_file" in st.session_state:
        del st.session_state["crv_file"]
    if "table_infinite_bus_file" in st.session_state:
        del st.session_state["table_infinite_bus_file"]

    if "base_case_simulation_data_df" in st.session_state:
        del st.session_state["base_case_simulation_data_df"]
    if "calibrated_simulation_data_df" in st.session_state:
        del st.session_state["calibrated_simulation_data_df"]
    if "custom_simulation_data_df" in st.session_state:
        del st.session_state["custom_simulation_data_df"]

    if "log_area_base_case" in st.session_state:
        del st.session_state["log_area_base_case"]
    if "log_area_base_case_content" in st.session_state:
        del st.session_state["log_area_base_case_content"]
    if "log_area_base_case_sensitivity" in st.session_state:
        del st.session_state["log_area_base_case_sensitivity"]
    if "log_area_base_case_sensitivity_content" in st.session_state:
        del st.session_state["log_area_base_case_sensitivity_content"]
    if "log_area_base_case_calibration" in st.session_state:
        del st.session_state["log_area_base_case_calibration"]
    if "log_area_base_case_calibration_content" in st.session_state:
        del st.session_state["log_area_base_case_calibration_content"]
    if "log_area_custom_calibration" in st.session_state:
        del st.session_state["log_area_custom_calibration"]
    if "log_area_custom_calibration_content" in st.session_state:
        del st.session_state["log_area_custom_calibration_content"]

    # Nettoyage des loggers
    clear_streamlit_logger("base_case_st_logger")
    clear_streamlit_logger("sensitivity_st_logger")
    clear_streamlit_logger("calibration_st_logger")
    clear_streamlit_logger("custom_calibration_st_logger")


def create_data_tab():
    col_1, col_2, col_3 = st.columns([4, 1, 8])

    if "last_zip_hash" not in st.session_state:
        st.session_state.last_zip_hash = None

    with col_1:
        zipped_data = st.file_uploader("Upload zip file", type="zip")

        if zipped_data is not None:
            current_hash = get_file_hash(zipped_data)

            if st.session_state.last_zip_hash != current_hash:
                st.session_state.last_zip_hash = current_hash

                temp_folders_base_cases = st.session_state["temp_folders_base_cases"] # Liste de dossiers des cas tests
                temp_folders_sensitivity_cases = st.session_state["temp_folders_sensitivity_cases"]
                temp_folders_calibration_cases = st.session_state["temp_folders_calibration_cases"]
                temp_folders_custom_calibration_cases = st.session_state["temp_folders_custom_calibration_cases"]

                # ATTENTION, à adapter pour plusieurs cas !
                reinit_analysis(
                    temp_folders_base_cases,
                    temp_folders_sensitivity_cases,
                    temp_folders_calibration_cases,
                    temp_folders_custom_calibration_cases
                )

                # cases_files is a dictionary : {'case_1' : [path to .txt, path to file .csv, ...], 'case_2':[...], ...}
                # common_par_file est le chemin d'un fichier.par reprennant les parametres communs aux cas à simuler et qu'on veut optimiser
                cases_files, common_par_file = extract_files(zipped_data, temp_folders_base_cases) 
                print("cases_files :", cases_files)
                
                # PMU data for P and Q (measured)
                measured_data_df_dic = {} # dictionnary of measured data frames by case
                sampled_measured_data_df_dic = {} # dictionnary of sampled measured data frames by case
                for case in cases_files:
                    uploaded_measured_file = [file for file in cases_files[case] if file.endswith(".csv")][0]
                    uploaded_measured_file = os.path.join(temp_folders_base_cases[case], uploaded_measured_file)
                    try:
                        measured_data_df = get_measured_data(uploaded_measured_file)
                        sampled_measured_data_df = sample_df(measured_data_df)
                        measured_data_df_dic[case] = measured_data_df
                        sampled_measured_data_df_dic[case] = sampled_measured_data_df
                    except Exception as e:
                        st.exception(e)

                st.session_state["measured_data_df_dic"] = measured_data_df_dic
                st.session_state["sampled_measured_data_df_dic"] = sampled_measured_data_df_dic

                # jobs file for Dynawo Simulation
                jobs_files_paths = {} # dictionnary of jobs files by case
                for case in cases_files:
                    uploaded_jobs = [file for file in cases_files[case] if file.endswith(".jobs")][0]
                    jobs_file_path = os.path.join(temp_folders_base_cases[case], uploaded_jobs)
                    jobs_files_paths[case] = jobs_file_path
                st.session_state["jobs_files"] = jobs_files_paths

                # # IIDM file for Dynawo Simulation
                # uploaded_iidms = [file for file in cases_files if file.endswith("iidm")]  # TODO: même traitement pour les autres ?
                # if len(uploaded_iidms) > 0:
                #     uploaded_iidm = uploaded_iidms[0]
                #     temp_iidm_file_path = os.path.join(temp_folders_base_cases, uploaded_iidm)
                #     st.session_state["iidm_file"] = temp_iidm_file_path

                # dyd file for Dynawo Simulation
                dyd_files_paths = {} # dictionnary of dyd files by case
                for case in cases_files:
                    uploaded_dyd = [file for file in cases_files[case] if file.endswith(".dyd")][0]
                    dyd_files_paths[case] = os.path.join(temp_folders_base_cases[case], uploaded_dyd)
                st.session_state["dyd_files"] = dyd_files_paths

                # par file for Dynawo Simulation
                par_files_paths = {} # dictionnary of par files by case
                for case in cases_files:  
                    uploaded_par = [file for file in cases_files[case] if file.endswith(".par")][0]
                    par_files_paths[case] = os.path.join(temp_folders_base_cases[case], uploaded_par)
                st.session_state["par_files"] = par_files_paths
                # something special related to the par file
                st.session_state["parameters_sets"] = get_parameters_sets(common_par_file)

                # crv file for Dynawo Simulation
                crv_files_paths = {} # dictionnary of crv files by case
                for case in cases_files:
                    uploaded_crv = [file for file in cases_files[case] if file.endswith(".crv")][0]
                    crv_files_paths[case] = os.path.join(temp_folders_base_cases[case], uploaded_crv)
                st.session_state["crv_files"] = crv_files_paths

                # U and Theta data at the infinite bus, for Dynawo Simulation
                infinite_bus_table_files_paths = {} # dictionnary of infinite_bus_table files by case
                for case in cases_files:
                    uploaded_infinite_bus_table = [file for file in cases_files[case] if file.endswith(".txt")][0]
                    infinite_bus_table_files_paths[case] = os.path.join(temp_folders_base_cases[case], uploaded_infinite_bus_table)
                st.session_state["infinite_bus_table_files"] = infinite_bus_table_files_paths

    with col_2:
        st.write("")  # for spacing

    with col_3:
        with st.expander("P (or U) and Q from PMU Data (measured_data.csv)"):
            if "measured_data_df_dic" in st.session_state:
                st.write(st.session_state["measured_data_df_dic"])
            else:
                st.write("Data have not been uploaded yet")
        with st.expander("Dynawo jobs files"):
            if "jobs_files" in st.session_state:
                st.write(st.session_state["jobs_files"])
            else:
                st.write("jobs files have not been uploaded yet")
        # with st.expander("Dynawo iidm files"):
        #     if "iidm_file" in st.session_state:
        #         uploaded_jobs_name = os.path.basename(st.session_state["iidm_file"])
        #         st.write("iidm files have been uploaded: " + uploaded_jobs_name)
        #     else:
        #         st.write("iidm files have not been uploaded yet")
        with st.expander("Dynawo dyd files"):
            if "dyd_files" in st.session_state:
                st.write(st.session_state["dyd_files"])
            else:
                st.write("dyd files have not been uploaded yet")
        with st.expander("Dynawo par files"):
            if "par_files" in st.session_state:
                st.write(st.session_state["par_files"])
            else:
                st.write("par files have not been uploaded yet")
        with st.expander("Dynawo crv files"):
            if "crv_files" in st.session_state:
                st.write(st.session_state["crv_files"])
            else:
                st.write("crv files have not been uploaded yet")
        with st.expander("Infinite bus table files"):
            if "infinite_bus_table_files" in st.session_state:
                st.write(st.session_state["infinite_bus_table_files"])
            else:
                st.write("Infinite bus table files have not been uploaded yet")


def extract_files(zipped_data, temp_folders_base_cases):
    
    """
    zipped_data : Data uploaded by Streamlit (zipfile)
    temp_folders_base_cases : list of 'base_case_i' absolute paths
        ex: [ ".../temp/base_case_1", ".../temp/base_case_2", ... ]
    
    The function expects the ZIP to contain folders:
        case_1/, case_2/, ..., matching the indices.
    """
    
    # Nombre de cas attendus
    nb_cases = len(temp_folders_base_cases)

    # Dictionnaire résultat
    cases_files = {}  # ex: {"case_1": {"measured_data": "...", ...}}
    common_par_file = None   # fichier .par commun

    # --- Extraction sécurisée ---
    with zipfile.ZipFile(zipped_data, "r") as zip_ref:

        for zip_info in zip_ref.infolist():

            print("fichier du zip :", zip_info.filename)

            # Ignorer les dossiers
            if zip_info.is_dir():
                continue

            # Exemple de zip_info.filename : "case_1/measured_data.csv"
            parts = zip_info.filename.split("/")
            print("parts :", parts)

            # Cas 1 : fichier common.par à la racine du zip
            if len(parts) == 1:
                filename = parts[0]

                # Nous voulons détecter un fichier .par commun
                if filename.endswith(".par"):
                    st.write(f"Fichier par commun détecté : {filename}")

                    # On l’extrait dans le dossier parent (celui de base_case_1)
                    parent_temp = os.path.dirname(list(temp_folders_base_cases.values())[0])
                    common_par_path = os.path.join(parent_temp, filename)

                    with zip_ref.open(zip_info) as src, open(common_par_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    common_par_file = common_par_path
                    continue

                else:
                    st.warning(f"Ignoring file at ZIP root: {filename}")
                    continue

            # Cas 2 : fichier dans case_i/ 
            case_name = parts[0]  # "case_1"
            inner_filename = "/".join(parts[1:])  # "measured_data.csv"

            # Vérifier que case_X est du bon format
            if not case_name.startswith("case_"):
                st.warning(f"Unexpected directory '{case_name}' in ZIP")
                continue

            # Convertir case_X → index X
            try:
                index = int(case_name.replace("case_", ""))
            except ValueError:
                st.error(f"Invalid case directory name: {case_name}")
                continue

            # Vérifier que l’index existe dans temp_folders_base_cases
            if index < 1 or index > nb_cases:
                st.error(f"Case index {index} found in ZIP but no base_case_{index} created.")
                continue

            # Dossier de destination
            dest_folder = temp_folders_base_cases[case_name]

            # Chemin final
            extracted_path = os.path.abspath(os.path.join(dest_folder, inner_filename))

            # Sécurité : éviter les traversals
            if not extracted_path.startswith(os.path.abspath(dest_folder)):
                st.warning(f"Skipping suspicious path in ZIP: {zip_info.filename}")
                continue

            # Créer dossier si besoin
            os.makedirs(os.path.dirname(extracted_path), exist_ok=True)

            # Extraire fichier
            with zip_ref.open(zip_info) as src, open(extracted_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            # Stocker pour vérif
            if case_name not in cases_files:
                cases_files[case_name] = []
            cases_files[case_name].append(extracted_path)

    # --- Vérifications obligatoires pour chaque case ---
    required_exts = [".csv", ".dyd", ".par", ".txt", ".jobs", ".crv"]

    for i in range(1, nb_cases + 1):
        case_name = f"case_{i}"

        if case_name not in cases_files:
            st.error(f"Missing folder {case_name} in ZIP.")
            continue

        found = cases_files[case_name]
        filenames = [os.path.basename(f) for f in found]

        for ext in required_exts:
            candidates = [f for f in filenames if f.endswith(ext)]
            if len(candidates) == 0:
                st.error(f"Missing *{ext}* file in ZIP for {case_name}.")
            elif len(candidates) > 1:
                st.error(f"Multiple *{ext}* files found in {case_name}: {candidates}")

    return cases_files, common_par_file


def are_measures_uploaded():
    if "measured_data_df" in st.session_state:
        return True
    else:
        return False


def are_dynawo_inputs_loaded():
    if "jobs_files" in st.session_state \
            and "dyd_files" in st.session_state \
            and "par_files" in st.session_state \
            and "crv_files" in st.session_state \
            and "infinite_bus_table_files" in st.session_state:
        return True
    else:
        return False


def is_base_case_calculated():
    if "base_case_simulation_data_df" in st.session_state:
        return True
    else:
        return False


def is_calibrated_case_calculated():
    if "calibrated_simulation_data_df" in st.session_state:
        return True
    else:
        return False

def is_custom_case_calculated():
    if "custom_simulation_data_df" in st.session_state:
        return True
    else:
        return False


def create_dynawo_tab():
    dynawo_launcher = st.session_state["dynawo_launcher"]
    
    if not are_dynawo_inputs_loaded():
        st.write("You need to upload the Dynawo input files before running a simulation")
        return
    
    jobs_files = st.session_state["jobs_files"]
    case_names = list(jobs_files.keys())
    print(jobs_files)

    # Zone logs
    if "log_area_dynawo" not in st.session_state:
        st.session_state["log_area_dynawo"] = st.empty()

    log_area = st.session_state["log_area_dynawo"]

    def on_click_run_all():
        
        # Dictionnaires qui stockeront les résultats multi-cas
        simulation_data = {}
        correlations = {}
        logs_dict = {}
        
        sampled_measured_data_df_dic = st.session_state["sampled_measured_data_df_dic"]

        log_area.code("Starting Dynawo batch simulation...\n", language="text")

        # -----------------------------------------------------------------
        #  Boucle sur tous les cas
        # -----------------------------------------------------------------
        for case_name in case_names:

            jobs_path = jobs_files[case_name]
            
            # Fenêtre temporelle de référence : les mesures
            measured_df = sampled_measured_data_df_dic[case_name]
            start_t = measured_df.index[0]
            end_t = measured_df.index[-1]

            logger_case = initialize_streamlit_logger(
                log_area,
                f"logger_{case_name}",
                debug=False
            )

            log_area.code(f"=== Running {case_name} ===\n", language="text")

            try:
                # Lancer Dynawo
                simu_df = run_dynawo(dynawo_launcher, jobs_path, logger_case)
                # Échantillonnage au même pas de temps que les mesures
                simu_resampled_df = sample_df(simu_df, start_t, end_t)
                # Corrélation
                corr_dict = create_correlation_dict(simu_resampled_df, measured_df)

                # Stockage dans dictionnaires cohérents
                simulation_data[case_name] = simu_df
                correlations[case_name] = corr_dict
                logs_dict[case_name] = get_streamlit_logs(logger_case)

            except DynawoFailedException:
                log_area.code(
                    f"[ERROR] Dynawo failed for {case_name}\n",
                    language="text"
                )
                logs_dict[case_name] = get_streamlit_logs(logger_case)
                # Continue avec les autres cas

        # -----------------------------------------------------------------
        #  Stockage final dans session_state
        # -----------------------------------------------------------------
        st.session_state["simulation_data_df"] = simulation_data
        st.session_state["correlation_dicts"] = correlations
        st.session_state["dynawo_logs"] = logs_dict

        log_area.code(
            "=== Dynawo batch simulation completed ===\n",
            language="text"
        )

    # ---------------------------------------------------------------------
    #  Bouton Run All Cases
    # ---------------------------------------------------------------------
    st.button(
        label=f"Run Dynawo Simulation for all {len(case_names)} cases",
        key="run_all_cases_button",
        type="primary",
        on_click=on_click_run_all
    )

    # ---------------------------------------------------------------------
    #  Réaffichage des logs si déjà présents (Streamlit reload)
    # ---------------------------------------------------------------------
    if "dynawo_logs" in st.session_state:
        logs = st.session_state["dynawo_logs"]
        txt = ""
        for case_name, content in logs.items():
            txt += f"### {case_name} ###\n{content}\n\n"
        log_area.code(txt)


def create_correlation_dict(sampled_calibrated_df, sampled_measured_data_df):
    correlation_dict = dict()

    col_name_p, col_name_q = get_col_name_p_q()

    print(col_name_p, col_name_q)
    print("col_name_p =", col_name_p)
    print("Simulation columns:", sampled_calibrated_df.columns)

    sampled_simulated_p = sampled_calibrated_df[col_name_p].values
    sampled_simulated_q = sampled_calibrated_df[col_name_q].values

    sampled_measured_p = sampled_measured_data_df[col_name_p].values
    sampled_measured_q = sampled_measured_data_df[col_name_q].values

    correlation_dict["rmse"] = rmse(
        sampled_measured_p, sampled_measured_q,
        sampled_simulated_p, sampled_simulated_q
    )
    corr_p = pearson_corr(sampled_measured_p, sampled_simulated_p)
    m_alpha_p, a_beta_p = similarity_metrics(sampled_measured_p, sampled_simulated_p)
    corr_q = pearson_corr(sampled_measured_q, sampled_simulated_q)
    m_alpha_q, a_beta_q = similarity_metrics(sampled_measured_q, sampled_simulated_q)
    correlation_dict["corr_p"] = corr_p
    correlation_dict["corr_q"] = corr_q
    correlation_dict["m_alpha_p"] = m_alpha_p
    correlation_dict["m_alpha_q"] = m_alpha_q
    correlation_dict["a_beta_p"] = a_beta_p
    correlation_dict["a_beta_q"] = a_beta_q

    return correlation_dict


def create_correlation_df(correlation_dict, index_label):
    corr_p = correlation_dict["corr_p"]
    m_alpha_p = correlation_dict["m_alpha_p"]
    a_beta_p = correlation_dict["a_beta_p"]
    corr_q = correlation_dict["corr_q"]
    m_alpha_q = correlation_dict["m_alpha_q"]
    a_beta_q = correlation_dict["a_beta_q"]
    rmse = correlation_dict["rmse"]

    similarity_p_df = pd.DataFrame(
        {
            "Correlation": [corr_p],
            "Mag metric": [m_alpha_p],
            "Ph metric": [a_beta_p]
        },
        index=[index_label]
    )
    similarity_p_df.index.name = "P"

    similarity_q_df = pd.DataFrame(
        {
            "Correlation": [corr_q],
            "Mag metric": [m_alpha_q],
            "Ph metric": [a_beta_q]
        },
        index=[index_label]
    )
    similarity_q_df.index.name = "Q"

    similarity_pq_df = pd.DataFrame(
        {
            "Correlation": [(corr_p + corr_q) / 2],
            "Mag metric": [(m_alpha_p + m_alpha_q) / 2],
            "Ph metric": [(a_beta_p + a_beta_q) / 2],
            "RMSE": [rmse]
        },
        index=[index_label]
    )
    similarity_pq_df.index.name = "Mean P and Q"

    return similarity_p_df, similarity_q_df, similarity_pq_df


def create_correlation_panel():
    if are_measures_uploaded() and is_base_case_calculated():
        base_case_correlation_dict = st.session_state["base_case_correlation_dict"]
        similarity_p_df, similarity_q_df, similarity_pq_df \
            = create_correlation_df(base_case_correlation_dict, "base_case vs measures")

        if is_calibrated_case_calculated():
            calibrated_correlation_dict = st.session_state["calibrated_correlation_dict"]
            new_row_p_df, new_row_q_df, new_row_pq_df \
                = create_correlation_df(calibrated_correlation_dict, "calibrated vs measures")

            similarity_p_df = pd.concat([similarity_p_df, new_row_p_df])
            similarity_p_df.index.name = "P"
            similarity_q_df = pd.concat([similarity_q_df, new_row_q_df])
            similarity_q_df.index.name = "Q"
            similarity_pq_df = pd.concat([similarity_pq_df, new_row_pq_df])
            similarity_pq_df.index.name = "Mean P and Q"

        if is_custom_case_calculated():
            custom_correlation_dict = st.session_state["custom_correlation_dict"]
            new_row_p_df, new_row_q_df, new_row_pq_df \
                = create_correlation_df(custom_correlation_dict, "custom vs measures")

            similarity_p_df = pd.concat([similarity_p_df, new_row_p_df])
            similarity_p_df.index.name = "P"
            similarity_q_df = pd.concat([similarity_q_df, new_row_q_df])
            similarity_q_df.index.name = "Q"
            similarity_pq_df = pd.concat([similarity_pq_df, new_row_pq_df])
            similarity_pq_df.index.name = "Mean P and Q"

        col_1, col_2, col_3 = st.columns([1, 1, 1])
        with col_1:
            st.write(similarity_p_df)
        with col_2:
            st.write(similarity_q_df)
        with col_3:
            st.write(similarity_pq_df)


def create_plots():
    fig = make_subplots(rows=1, cols=2, subplot_titles=("P", "Q"))

    col_name_p, col_name_q = get_col_name_p_q()

    if are_measures_uploaded():
        measured_data_df = st.session_state["measured_data_df"]
        x = measured_data_df.index
        y_p = measured_data_df[col_name_p].values
        y_q = measured_data_df[col_name_q].values
        fig.add_trace(
            go.Scatter(x=x, y=y_p, mode='lines', name='P_measured', line=dict(color='magenta', width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y_q, mode='lines', name='Q_measured', line=dict(color='magenta', width=1.5)),
            row=1, col=2
        )
    else:
        st.write("you need to upload the measured data first")

    if is_base_case_calculated():
        base_case_simulation_data_df = st.session_state["base_case_simulation_data_df"]
        x = base_case_simulation_data_df.index
        y_p = base_case_simulation_data_df[col_name_p].values
        y_q = base_case_simulation_data_df[col_name_q].values
        fig.add_trace(
            go.Scatter(x=x, y=y_p, mode='lines', name='P_base_case', line=dict(color='blue', width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y_q, mode='lines', name='Q_base_case', line=dict(color='blue', width=1.5)),
            row=1, col=2
        )

    if is_calibrated_case_calculated():
        calibrated_simulation_data_df = st.session_state["calibrated_simulation_data_df"]
        x = calibrated_simulation_data_df.index
        y_p = calibrated_simulation_data_df[col_name_p].values
        y_q = calibrated_simulation_data_df[col_name_q].values
        fig.add_trace(
            go.Scatter(x=x, y=y_p, mode='lines', name='P_automatic_calibration', line=dict(color='green', width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y_q, mode='lines', name='Q_Automatic_calibration', line=dict(color='green', width=1.5)),
            row=1, col=2
        )

    if is_custom_case_calculated():
        custom_simulation_data_df = st.session_state["custom_simulation_data_df"]
        x = custom_simulation_data_df.index
        y_p = custom_simulation_data_df[col_name_p].values
        y_q = custom_simulation_data_df[col_name_q].values
        fig.add_trace(
            go.Scatter(x=x, y=y_p, mode='lines', name='P_custom_calibration', line=dict(color='cyan', width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y_q, mode='lines', name='Q_custom_calibration', line=dict(color='cyan', width=1.5)),
            row=1, col=2
        )

    st.plotly_chart(fig, use_container_width=True)


def create_plot_tab():
    create_correlation_panel()
    create_plots()


def create_sensitivity_tab():
    if is_base_case_calculated():
        col_1, col_2, col_3, col_4, col_5 = st.columns([3, 1, 5, 1, 3])
        with col_1:
            st.write("Select parameters to analyze during the sensitivity analysis")
            parameters_sets = st.session_state["parameters_sets"]
            selected_sets = dict()
            for set_id in list(parameters_sets.keys()):
                with st.expander(set_id):
                    selected_params = dict()
                    for parameter_id in parameters_sets[set_id]:
                        if st.checkbox(parameter_id, key="sensitivity_" + set_id + "_" + parameter_id, value=False):
                            selected_params[parameter_id] = parameters_sets[set_id][parameter_id]
                if len(selected_params) > 0:
                    selected_sets[set_id] = selected_params

        with col_2:
            st.write("")  # for spacing
        with col_3:

            def on_click():
                streamlit_logger_sensitivity = initialize_streamlit_logger(
                    st.session_state["log_area_base_case_sensitivity"], "sensitivity_st_logger", debug=False)

                if len(selected_sets) > 0:
                    temp_folders_sensitivity_cases = st.session_state["temp_folders_sensitivity_cases"]
                    for element in os.listdir(st.session_state["temp_folders_base_cases"]):
                        src = os.path.join(st.session_state["temp_folders_base_cases"], element)
                        if os.path.isfile(src):
                            shutil.copy(src, temp_folders_sensitivity_cases)
                    jobs_file_sensitivity = os.path.join(
                        temp_folders_sensitivity_cases,
                        os.path.basename(st.session_state["jobs_file"])
                    )
                    par_file_sensitivity = os.path.join(
                        temp_folders_sensitivity_cases,
                        os.path.basename(st.session_state["par_file"])
                    )

                    sampled_measured_data_df = st.session_state["sampled_measured_data_df"]
                    col_name_p, col_name_q = get_col_name_p_q()
                    sampled_measured_p = sampled_measured_data_df[col_name_p].values
                    sampled_measured_q = sampled_measured_data_df[col_name_q].values
                    base_case_rmse = st.session_state["base_case_correlation_dict"]["rmse"]

                    dynawo_launcher = st.session_state["dynawo_launcher"]
                    sensitivities = run_sensitivity_analysis(
                        dynawo_launcher,
                        jobs_file_sensitivity,
                        par_file_sensitivity,
                        selected_sets,
                        sampled_measured_p,
                        sampled_measured_q,
                        base_case_rmse,
                        streamlit_logger_sensitivity
                    )
                    st.session_state["sensitivities"] = sensitivities
                    st.session_state["log_area_base_case_sensitivity_content"] = get_streamlit_logs(streamlit_logger_sensitivity)
                else:
                    st.session_state["log_area_base_case_sensitivity_content"] = "You should select at least one parameter"

            st.button(
                label="Run Sensitivity Analysis",
                key="sensitivity_analysis_button",
                type="primary",
                on_click=on_click
            )

            if "log_area_base_case_sensitivity" not in st.session_state:
                st.session_state["log_area_base_case_sensitivity"] = st.empty()
            else:
                log_area_base_case_sensitivity = st.session_state["log_area_base_case_sensitivity"]
                if "log_area_base_case_sensitivity_content" in st.session_state:
                    log_area_base_case_sensitivity.code(st.session_state["log_area_base_case_sensitivity_content"])

        with col_4:
            st.write("")  # for spacing
        with col_5:
            if "sensitivities" in st.session_state:
                for set_id in st.session_state["sensitivities"]:
                    st.write(set_id)
                    sensitivity_df = st.session_state["sensitivities"][set_id]
                    st.write(sensitivity_df)
    else:
        st.write("You need to run a Dynawo simulation on the base_case first")


def create_param_calibration_tab():
    if is_base_case_calculated():
        col_1, col_2, col_3 = st.columns([8, 1, 8])
        with col_1:
            st.write("Select parameters to calibrate")
            parameters_sets = st.session_state["parameters_sets"]
            selected_sets = dict()
            for set_id in list(parameters_sets.keys()):
                with st.expander(set_id):
                    selected_params = dict()
                    for parameter_id in parameters_sets[set_id]:
                        parameter = parameters_sets[set_id][parameter_id]
                        param_type = parameter.get_type()
                        if param_type == "DOUBLE":
                            reference_value = float(parameter.get_reference_value())
                            min_bound = parameter.get_min_bound()
                            max_bound = parameter.get_max_bound()
                            col_checkbox, col_initial_value, col_min_bound, col_max_bound = st.columns([2, 1, 1, 1])
                            with col_checkbox:
                                if st.checkbox(parameter_id, key="automatic_calibration_" + set_id + "_" + parameter_id, value=False):
                                    selected_params[parameter_id] = parameter
                            with col_initial_value:
                                if parameter_id in selected_params:
                                    st.text_input("Reference value", value=reference_value,
                                                  key="automatic_calibration_initial_" + set_id + "_" + parameter_id, disabled=True)
                            with col_min_bound:
                                if parameter_id in selected_params:
                                    min_bound = st_keyup("Min bound", value=min_bound,
                                                     key="automatic_calibration_min_bound_" + set_id + "_" + parameter_id)
                                    parameter.set_min_bound(min_bound)
                            with col_max_bound:
                                if parameter_id in selected_params:
                                    max_bound = st_keyup("Max bound", value=max_bound,
                                                     key="automatic_calibration_max_bound_" + set_id + "_" + parameter_id)
                                    parameter.set_max_bound(max_bound)
                        else:
                            continue

                if len(selected_params) > 0:
                    selected_sets[set_id] = selected_params
        with col_2:
            st.write("")  # for spacing
        with col_3:
            with st.expander("Optimization parameters (advanced)"):
                optim_method = st.radio(
                    "Select optimization method",
                    [OptimMethod.NELDER_MEAD.value, OptimMethod.DIFFERENTIAL_EVOLUTION.value],
                    index=0
                )
                optim_method = OptimMethod(optim_method)

            def on_click():
                streamlit_logger_calibration = initialize_streamlit_logger(
                    st.session_state["log_area_base_case_calibration"], "calibration_st_logger", debug=False)

                if len(selected_sets) > 0:
                    temp_folders_calibration_cases = st.session_state["temp_folders_calibration_cases"]
                    for element in os.listdir(st.session_state["temp_folders_base_cases"]):
                        src = os.path.join(st.session_state["temp_folders_base_cases"], element)
                        if os.path.isfile(src):
                            shutil.copy(src, temp_folders_calibration_cases)
                    jobs_file_calibration = os.path.join(
                        temp_folders_calibration_cases,
                        os.path.basename(st.session_state["jobs_file"])
                    )
                    par_file_calibration = os.path.join(
                        temp_folders_calibration_cases,
                        os.path.basename(st.session_state["par_file"])
                    )

                    sampled_measured_data_df = st.session_state["sampled_measured_data_df"]
                    col_name_p, col_name_q = get_col_name_p_q()
                    sampled_measured_p = sampled_measured_data_df[col_name_p].values
                    sampled_measured_q = sampled_measured_data_df[col_name_q].values

                    base_case_rmse = st.session_state["base_case_correlation_dict"]["rmse"]

                    dynawo_launcher = st.session_state["dynawo_launcher"]

                    try:
                        calibrated_simulation_data_df = run_parameter_calibration(
                            dynawo_launcher,
                            jobs_file_calibration,
                            par_file_calibration,
                            selected_sets,
                            sampled_measured_p,
                            sampled_measured_q,
                            base_case_rmse,
                            optim_method,
                            streamlit_logger=streamlit_logger_calibration
                        )

                        sampled_measured_data_df = st.session_state["sampled_measured_data_df"]
                        measured_data_start_time = sampled_measured_data_df.index[0]
                        measured_data_end_time = sampled_measured_data_df.index[-1]
                        sampled_calibrated_simulation_data_df = sample_df(
                            calibrated_simulation_data_df,
                            measured_data_start_time,
                            measured_data_end_time
                        )

                        calibrated_correlation_dict = create_correlation_dict(sampled_calibrated_simulation_data_df)

                        st.session_state["calibrated_simulation_data_df"] = calibrated_simulation_data_df
                        st.session_state["calibrated_correlation_dict"] = calibrated_correlation_dict
                        st.session_state["log_area_base_case_calibration_content"] = get_streamlit_logs(streamlit_logger_calibration)
                    except Exception as e:  # Dynawo doesn't converge, or problem with bound inconsistency
                        if "calibrated_simulation_data_df" in st.session_state:
                            del st.session_state["calibrated_simulation_data_df"]
                        if "calibrated_correlation_dict" in st.session_state:
                            del st.session_state["calibrated_correlation_dict"]
                        st.session_state["log_area_base_case_calibration_content"] = get_streamlit_logs(streamlit_logger_calibration)
                        streamlit_logger_calibration.error(e)
                else:
                    st.session_state["log_area_base_case_calibration_content"] = "You should select at least one parameter"

            st.button(
                label="Run Calibration",
                key="calibration_button",
                type="primary",
                on_click=on_click
            )

            if "log_area_base_case_calibration" not in st.session_state:
                st.session_state["log_area_base_case_calibration"] = st.empty()
            else:
                log_area_base_case_calibration = st.session_state["log_area_base_case_calibration"]
                if "log_area_base_case_calibration_content" in st.session_state:
                    log_area_base_case_calibration.code(st.session_state["log_area_base_case_calibration_content"])

    else:
        st.write("You need to run a Dynawo simulation on the base_case first")


def create_custom_calibration_tab():
    if is_base_case_calculated():
        col_1, col_2, col_3 = st.columns([8, 1, 8])
        with col_1:
            st.write("Select parameters to calibrate")
            parameters_sets = st.session_state["parameters_sets"]
            selected_sets = dict()
            for set_id in list(parameters_sets.keys()):
                with st.expander(set_id):
                    selected_params = dict()
                    for parameter_id in parameters_sets[set_id]:
                        parameter = parameters_sets[set_id][parameter_id]
                        col_checkbox, col_initial_value, col_input = st.columns([2, 1, 1])
                        with col_checkbox:
                            if st.checkbox(parameter_id, key="custom_calibration_" + set_id + "_" + parameter_id, value=False):
                                selected_params[parameter_id] = parameter
                        with col_initial_value:
                            if parameter_id in selected_params:
                                st.text_input("Reference value", value=parameter.get_reference_value(),
                                              key="custom_calibration_initial_" + set_id + "_" + parameter_id, disabled=True)
                        with col_input:
                            if parameter_id in selected_params:
                                new_value = st_keyup("New value", value=parameter.get_value(),
                                                 key="custom_calibration_input_" + set_id + "_" + parameter_id)
                                parameter.set_value(new_value)
                                selected_params[parameter_id] = parameter
                if len(selected_params) > 0:
                    selected_sets[set_id] = selected_params
        with col_2:
            st.write("")  # for spacing
        with col_3:

            def on_click():
                streamlit_logger_custom_calibration = initialize_streamlit_logger(
                    st.session_state["log_area_custom_calibration"], "custom_calibration_st_logger", debug=False)

                if len(selected_sets) > 0:
                    temp_folders_custom_calibration_cases = st.session_state["temp_folders_custom_calibration_cases"]
                    for element in os.listdir(st.session_state["temp_folders_base_cases"]):
                        src = os.path.join(st.session_state["temp_folders_base_cases"], element)
                        if os.path.isfile(src):
                            shutil.copy(src, temp_folders_custom_calibration_cases)
                    jobs_file_calibration = os.path.join(
                        temp_folders_custom_calibration_cases,
                        os.path.basename(st.session_state["jobs_file"])
                    )
                    par_file_calibration = os.path.join(
                        temp_folders_custom_calibration_cases,
                        os.path.basename(st.session_state["par_file"])
                    )

                    dynawo_launcher = st.session_state["dynawo_launcher"]
                    try:
                        custom_simulation_data_df = run_custom_dynawo(
                            dynawo_launcher,
                            jobs_file_calibration,
                            par_file_calibration,
                            selected_sets,
                            streamlit_logger=streamlit_logger_custom_calibration
                        )

                        sampled_measured_data_df = st.session_state["sampled_measured_data_df"]
                        measured_data_start_time = sampled_measured_data_df.index[0]
                        measured_data_end_time = sampled_measured_data_df.index[-1]
                        sampled_custom_simulation_data_df = sample_df(
                            custom_simulation_data_df,
                            measured_data_start_time,
                            measured_data_end_time
                        )

                        custom_correlation_dict = create_correlation_dict(sampled_custom_simulation_data_df)

                        st.session_state["custom_simulation_data_df"] = custom_simulation_data_df
                        st.session_state["custom_correlation_dict"] = custom_correlation_dict
                        st.session_state["log_area_custom_calibration_content"] = get_streamlit_logs(streamlit_logger_custom_calibration)
                    except DynawoFailedException:
                        if "custom_simulation_data_df" in st.session_state:
                            del st.session_state["custom_simulation_data_df"]
                        if "custom_correlation_dict" in st.session_state:
                            del st.session_state["custom_correlation_dict"]
                        st.session_state["log_area_custom_calibration_content"] = get_streamlit_logs(
                            streamlit_logger_custom_calibration)
                else:
                    st.session_state["log_area_custom_calibration_content"] = "You should select at least one parameter"

            st.button(
                label="Run Dynawo Simulation with custom parameters",
                key="custom_calibration_button",
                type="primary",
                on_click=on_click
            )

            if "log_area_custom_calibration" not in st.session_state:
                st.session_state["log_area_custom_calibration"] = st.empty()
            else:
                log_area_custom_calibration = st.session_state["log_area_custom_calibration"]
                if "log_area_custom_calibration_content" in st.session_state:
                    log_area_custom_calibration.code(st.session_state["log_area_custom_calibration_content"])
    else:
        st.write("You need to run a Dynawo simulation on the base_case first")


def main():

    # Creating the webapp
    create_webpage_header()

    # Initialize the settings with default values
    settings = Settings()
    dynawo_launcher = settings.get_dynawo_launcher()

    #Stockage launcher dynawo dans streamlit
    st.session_state["dynawo_launcher"] = dynawo_launcher
    
    #Detection repertoire courant
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # -------- CONFIGURER ICI LE NOMBRE DE SCENARIOS PRIS EN CHARGE --------
    nb_cases = 2  # Mets ici le nombre que tu veux (2, 3, ...)
    
    #Creation dossiers temporaires
    temp_folder = os.path.join(script_directory, "..", "temp")
    folders_to_create = [temp_folder]
    temp_folders_base_cases = {}
    temp_folders_sensitivity_cases = {}
    temp_folders_calibration_cases = {}
    temp_folders_custom_calibration_cases = {}

    for i in range(nb_cases):
        base = os.path.join(temp_folder, f"base_case_{i+1}")
        temp_folders_base_cases[f"case_{i+1}"] = base
        sensitivity = os.path.join(temp_folder, f"sensitivity_case_{i+1}")
        temp_folders_sensitivity_cases[f"case_{i+1}"] = sensitivity
        calibration = os.path.join(temp_folder, f"calibration_case_{i+1}")
        temp_folders_calibration_cases[f"case_{i+1}"] = calibration
        custom = os.path.join(temp_folder, f"custom_case_{i+1}")
        temp_folders_custom_calibration_cases[f"case_{i+1}"] = custom
        folders_to_create += [base, sensitivity, calibration, custom]

    #Creation des dossiers s'il sont absents
    for folder in folders_to_create:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    #Stockage des chemins dans session state
    st.session_state["temp_folders_base_cases"] = temp_folders_base_cases # dictionnaire {'case_1' : 'path to base_case temporary directory', 'case_2' : ..., ...}
    st.session_state["temp_folders_sensitivity_cases"] = temp_folders_sensitivity_cases
    st.session_state["temp_folders_calibration_cases"] = temp_folders_calibration_cases
    st.session_state["temp_folders_custom_calibration_cases"] = temp_folders_custom_calibration_cases
    st.session_state["nb_cases"] = nb_cases
    print(temp_folders_base_cases)

    # Creating tabs
    data_tab, dynawo_tab, plot_tab, sensitivity_tab, param_calibration_tab, custom_param_calibration_tab \
        = st.tabs(
        [
            "Upload Data",
            "Run Dynawo Simulation",
            "Visualize Plots",
            "Sensitivity Analysis",
            "Automatic Parameter Calibration",
            "Custom Parameter Calibration"
         ]
    )
    with data_tab:
        create_data_tab()
    with dynawo_tab:
        create_dynawo_tab()
    with plot_tab:
        create_plot_tab()
    with sensitivity_tab:
        create_sensitivity_tab()
    with param_calibration_tab:
        create_param_calibration_tab()
    with custom_param_calibration_tab:
        create_custom_calibration_tab()


if __name__ == "__main__":
    main()
