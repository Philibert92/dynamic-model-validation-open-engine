# Copyright (c) 2022-2025, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the dynamic-model-validation-engine project.

"""
This file contains various methods to work on the data.
"""

import os
import shutil
import hashlib
import pandas as pd
import numpy as np
import math
import streamlit as st
from scipy.fft import fft
from scipy.stats import pearsonr


def get_measured_data(csv_path):
    measured_data_df = pd.read_csv(csv_path, delimiter=",")
    measured_data_df.set_index("time", inplace=True)
    measured_data_df = remove_rows_with_same_index(measured_data_df)
    col_name_p = [col for col in measured_data_df.columns
                  if col.endswith("PGenPu") or col.endswith("PGenNomPu") or col.endswith("PPu") or col.endswith("UPu")][0]
    col_name_q = [col for col in measured_data_df.columns
                  if col.endswith("QGenPu") or col.endswith("QGenNomPu") or col.endswith("QPu")][0]
    st.session_state["col_name_p"] = col_name_p
    st.session_state["col_name_q"] = col_name_q
    measured_data_df[col_name_p] = pd.to_numeric(measured_data_df[col_name_p], errors='coerce')
    measured_data_df[col_name_q] = pd.to_numeric(measured_data_df[col_name_q], errors='coerce')

    return measured_data_df


def get_col_name_p_q():
    col_name_p = st.session_state["col_name_p"] if "col_name_p" in st.session_state else None
    col_name_q = st.session_state["col_name_q"] if "col_name_q" in st.session_state else None
    return col_name_p, col_name_q


def remove_rows_with_same_index(df):
    return df[~df.index.duplicated(keep='first')]


def sample_df(df, start_time=None, end_time=None, sampling_frequency=100):
    """
    start_time: measure start time
    end_time: end_time end time

    The indices of the sampled df of simulation data must match with the indices of the measured data.
    """
    df = df.sort_index()

    if start_time == None:
        start_time = df.index[0]
    else:
        if start_time < df.index[0]:
            raise ValueError(f"start_time {start_time} is before the minimum index in df ({df.index[0]})")
    if end_time == None:
        end_time = df.index[-1]
    else:
        if end_time > df.index[-1]:
            raise ValueError(f"end_time {end_time} is after the maximum index in df ({df.index[-1]})")

    timestep = 1 / sampling_frequency
    sampled_time_indices = np.linspace(start_time, end_time, num=int(np.round((end_time - start_time) / timestep)) + 1)
    sampled_df = pd.DataFrame(index=sampled_time_indices, columns=df.columns)
    for sampled_time_index in sampled_time_indices:
        if sampled_time_index in df.index:
            sampled_df.loc[sampled_time_index] = df.loc[sampled_time_index]
        else:
            lower_index = max(df.index[df.index < sampled_time_index])
            upper_index = min(df.index[df.index > sampled_time_index])
            position = (sampled_time_index - lower_index) / (upper_index - lower_index)
            interpolated_row = (1 - position) * df.loc[lower_index] + position * df.loc[upper_index]
            sampled_df.loc[sampled_time_index] = interpolated_row
    return sampled_df


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    correlation, _ = pearsonr(x, y)
    return correlation


def similarity_metrics(x, y):
    """
    Metric proposed in
    "An alternative approach to measure similarity between two deterministic transient signals",
    Kihong Shin

    x and y are numpy Arrays of same length
    """
    alpha = 20
    beta = 1
    fft_x = fft(x)
    fft_y = fft(y)
    h = [fft_x[i] / fft_y[i] for i in range(len(fft_x))]
    mag_h = np.absolute(h)
    phi_h = np.angle(h)
    db = np.absolute(20 * np.log10(mag_h) / alpha)
    angle_ratio = np.absolute(phi_h / (2 * math.pi * beta))
    m_alpha = 1 - np.mean(np.tanh(np.log(3) / 2 * db))
    a_beta = 1 - np.mean(np.tanh(np.log(3) / 2 * angle_ratio))
    return m_alpha, a_beta


def rmse(measured_p, measured_q, simulated_p, simulated_q):
    """
    measured_p, measured_q, simulated_p, simulated_q are numpy Arrays of same length
    (i.e. after sampling)
    """
    errors = np.square(measured_p - simulated_p) + np.square(measured_q - simulated_q)
    mean_error = np.mean(errors)
    rmse = np.sqrt(mean_error)
    return rmse


def clean_directory(folder_path):
    for file_object in os.listdir(folder_path):
        file_object_path = os.path.join(folder_path, file_object)
        if os.path.isfile(file_object_path) or os.path.islink(file_object_path):
            os.unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)


def get_file_hash(file):
    file.seek(0)
    file_hash = hashlib.sha256(file.read()).hexdigest()
    file.seek(0)
    return file_hash

