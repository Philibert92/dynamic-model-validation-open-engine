# Copyright (c) 2022-2025, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the dynamic-model-validation-engine project.

"""
This file contains the functions that enables to run dynawo and
to modify the parameter values of the inputs.
"""

import os
import pandas as pd
import subprocess
from settings import Settings
from util_functions import remove_rows_with_same_index, sample_df, rmse
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy


class DynawoFailedException(Exception):
    pass


class DynawoParam:
    def __init__(self, name, type, reference_value):
        self._name = name
        self._type = type  # "INT", "DOUBLE", "STRING", ...
        self._reference_value = reference_value  # string
        self._value = reference_value  # string
        if type == "DOUBLE":
            # if param = 4, we search in the range [-4, 40]
            # in order to catch a sign error or an error in the order of magnitude
            # The larger the range the longer the computation time
            reference_value = float(reference_value)
            self._min_bound = min(-reference_value, 10 * reference_value)
            self._max_bound = max(-reference_value, 10 * reference_value)
        else:
            self._min_bound = None
            self._max_bound = None

    def get_name(self):
        return self._name

    def get_type(self):
        return self._type

    def get_reference_value(self):
        return self._reference_value

    def get_value(self):
        return self._value

    def get_min_bound(self):
        return self._min_bound

    def get_max_bound(self):
        return self._max_bound

    def set_name(self, value):
        self._name = value

    def set_type(self, value):
        self._type = value

    def set_reference_value(self, value):
        self._reference_value = value

    def set_value(self, value):
        self._value = value

    def set_min_bound(self, value):
        try:
            value = float(value)
            self._min_bound = value
        except ValueError:
            self._min_bound = None

    def set_max_bound(self, value):
        try:
            value = float(value)
            self._max_bound = value
        except ValueError:
            self._max_bound = None


def run_dynawo(dynawo_launcher, jobs_file, streamlit_logger=None):
    """
    The current version of this function calls dynawo via subprocess.
    In the future it might be rewritten by calling dynawo directly from pypowsybl
    """
    init_wd = os.getcwd()
    input_wd = os.path.dirname(jobs_file)

    # With his implementation Dynawo should be called from the folder
    # which contains the input files.
    os.chdir(input_wd)

    cmd = [dynawo_launcher, "jobs", jobs_file]

    if streamlit_logger is not None:
        streamlit_logger.info("Running Dynawo: " + ' '.join(cmd))
    t = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = t.communicate()
    return_code = t.returncode
    if streamlit_logger is not None:
        if return_code != 0:
            streamlit_logger.error("Dynawo simulation failed!")
            msg = stderr.decode("utf-8")
            streamlit_logger.error(msg)
        else:
            streamlit_logger.info("Dynawo simulation completed")

    if return_code != 0:
        os.chdir(init_wd)
        raise DynawoFailedException

    os.chdir(init_wd)

    simulation_data_df = get_simulation_data(jobs_file)

    return simulation_data_df


def modify_one_param_par_file(par_file, set_id, param_id, new_value):
    tree = ET.parse(par_file)
    root = tree.getroot()
    namespace = {'ns': 'http://www.rte-france.com/dynawo'}
    for par in root.findall('.//ns:set[@id="{}"]/ns:par[@name="{}"]'.format(set_id, param_id), namespace):
        par.set('value', str(new_value))
    ET.register_namespace('', 'http://www.rte-france.com/dynawo')
    modified_content = ET.tostring(root, encoding='unicode')
    with open(par_file, 'w') as file:
        file.write(modified_content)


def modify_multiple_param_par_file(par_file, selected_sets):
    """
    selected_sets: dict(set_id, dict(param_id, new_value))
    """
    tree = ET.parse(par_file)
    root = tree.getroot()
    namespace = {'ns': 'http://www.rte-france.com/dynawo'}
    for set_id in selected_sets:
        for parameter_id in selected_sets[set_id]:
            parameter = selected_sets[set_id][parameter_id]
            new_value = parameter.get_value()
            xpath = './/ns:set[@id="{}"]/ns:par[@name="{}"]'.format(set_id, parameter_id)
            for par in root.findall(xpath, namespace):
                par.set('value', str(new_value))
    ET.register_namespace('', 'http://www.rte-france.com/dynawo')
    modified_content = ET.tostring(root, encoding='unicode')
    with open(par_file, 'w') as file:
        file.write(modified_content)


def modify_multiple_param_par_file_for_optim(par_files, x, index_to_param_map):
    """
    Modifie les paramètres spécifiés dans tous les fichiers .par donnés dans par_files.
    
    par_files : dict {case_name: path_to_par_file}
    x : liste ou array des valeurs de paramètres (par ex résultat de l'optimiseur)
    index_to_param_map : dict {index: (set_id, param_name)}
    """

    namespace = {'ns': 'http://www.rte-france.com/dynawo'}
    ET.register_namespace('', 'http://www.rte-france.com/dynawo')

    for case_name, par_file in par_files.items():
        tree = ET.parse(par_file)
        root = tree.getroot()

        # Modification des paramètres communs à tous les fichiers
        for idx, (set_id, param_name) in index_to_param_map.items():
            
            param_value = x[idx]

            print('CHANGING PARAMETER : ', idx, set_id, param_name, 'WITH VALUE : ', param_value)

            xpath = f'.//ns:set[@id="{set_id}"]/ns:par[@name="{param_name}"]'

            for par in root.findall(xpath, namespace):
                par.set('value', str(param_value))

        # Enregistrement du fichier modifié
        modified_content = ET.tostring(root, encoding='unicode')
        with open(par_file, 'w') as file:
            file.write(modified_content)


def run_custom_dynawo(
        dynawo_launcher,
        jobs_file,
        par_file,
        selected_sets,
        streamlit_logger=None
):
    if streamlit_logger is not None:
        for set_id in selected_sets:
            for parameter_id in selected_sets[set_id]:
                parameter = selected_sets[set_id][parameter_id]
                new_value = parameter.get_value()
                streamlit_logger.info(set_id + "/" + parameter.get_name() + " - " + "new_value: " + str(new_value))
    modify_multiple_param_par_file(par_file, selected_sets)
    simulation_data_df = run_dynawo(dynawo_launcher, jobs_file, streamlit_logger)
    return simulation_data_df


def create_output_folder(output_folder=None):
    if output_folder is None:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_directory, "temp")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    return output_folder


def get_simulation_data(jobs_file):
    tree = ET.parse(jobs_file)
    root = tree.getroot()
    out_csv = os.path.join(
        os.path.dirname(jobs_file),
        root.find("{*}job/{*}outputs").get("directory"),
        "curves",
        "curves.csv"
    )
    simulation_data_df = pd.read_csv(out_csv, delimiter=";")
    simulation_data_df.set_index("time", inplace=True)
    simulation_data_df = remove_rows_with_same_index(simulation_data_df)
    return simulation_data_df


def get_parameters_sets(par_file):
    tree = ET.parse(par_file)
    root = tree.getroot()
    parameters_sets = dict()
    for set_element in root:
        set_id = set_element.attrib["id"]
        parameters = dict()
        for param_element in set_element:
            if param_element.tag.endswith("par"):
                param_name = param_element.attrib["name"]
                param_type = param_element.attrib["type"]
                param_value = param_element.attrib["value"]
                parameter = DynawoParam(param_name, param_type, param_value)
                parameters[param_name] = parameter
            # there exists also "reference" tags, but they refer to init values in the iidm file
            # these values are not relevant for the purpose of this model validation application
        parameters_sets[set_id] = parameters
    print("parameters_set :", parameters_sets)
    return parameters_sets


if __name__ == "__main__":

    output_folder = create_output_folder()
    settings = Settings()
    dynawo_launcher = settings.get_dynawo_launcher()

    example_jobs_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "test_cases", "test_case_1", "test_case_1.jobs")
    try:
        simulation_data_df = run_dynawo(dynawo_launcher, example_jobs_file)
        simulation_data_df = sample_df(simulation_data_df)
        print(simulation_data_df)
    except DynawoFailedException:
        print("Error with the dynawo computation")
