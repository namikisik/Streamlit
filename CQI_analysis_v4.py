#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jan 11 10:04:58 2021

@author: macbookpro
"""

import pandas as pd
import streamlit as st
from PIL import Image
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#%% 
#reading CQI file
# @st.cache
def _cqi_average(row): #cqi calculation
    try:
        return ((row['[H] L.ChMeas.CQI.DL.0'] * 0) + (row['[H] L.ChMeas.CQI.DL.1'] * 1)+
               (row['[H] L.ChMeas.CQI.DL.2'] * 2) + (row['[H] L.ChMeas.CQI.DL.3'] * 3) +
               (row['[H] L.ChMeas.CQI.DL.4'] * 4) + (row['[H] L.ChMeas.CQI.DL.5'] * 5) +
               (row['[H] L.ChMeas.CQI.DL.6'] * 6) + (row['[H] L.ChMeas.CQI.DL.7'] * 7) +
               (row['[H] L.ChMeas.CQI.DL.8'] * 8) + (row['[H] L.ChMeas.CQI.DL.9'] * 9) +
               (row['[H] L.ChMeas.CQI.DL.10'] * 10) + (row['[H] L.ChMeas.CQI.DL.11'] * 11) +
               (row['[H] L.ChMeas.CQI.DL.12'] * 12) + (row['[H] L.ChMeas.CQI.DL.13'] * 13) +
               (row['[H] L.ChMeas.CQI.DL.14'] * 14) + (row['[H] L.ChMeas.CQI.DL.15'] * 15)
               ) / (row['[H] L.ChMeas.CQI.DL.0'] + row['[H] L.ChMeas.CQI.DL.1'] +
               row['[H] L.ChMeas.CQI.DL.2'] + row['[H] L.ChMeas.CQI.DL.3'] +
               row['[H] L.ChMeas.CQI.DL.4'] + row['[H] L.ChMeas.CQI.DL.5'] +
               row['[H] L.ChMeas.CQI.DL.6'] + row['[H] L.ChMeas.CQI.DL.7'] +
               row['[H] L.ChMeas.CQI.DL.8'] + row['[H] L.ChMeas.CQI.DL.9'] +
               row['[H] L.ChMeas.CQI.DL.10'] + row['[H] L.ChMeas.CQI.DL.11'] +
               row['[H] L.ChMeas.CQI.DL.12'] + row['[H] L.ChMeas.CQI.DL.13'] +
               row['[H] L.ChMeas.CQI.DL.14'] + row['[H] L.ChMeas.CQI.DL.15'])
    except ZeroDivisionError:
        return 0


cqi_cols = ['DATETIME', 'CELL', '[H] L.ChMeas.CQI.DL.0',
            '[H] L.ChMeas.CQI.DL.1', '[H] L.ChMeas.CQI.DL.2',
            '[H] L.ChMeas.CQI.DL.3','[H] L.ChMeas.CQI.DL.4',
            '[H] L.ChMeas.CQI.DL.5', '[H] L.ChMeas.CQI.DL.6',
            '[H] L.ChMeas.CQI.DL.7', '[H] L.ChMeas.CQI.DL.8',
            '[H] L.ChMeas.CQI.DL.9', '[H] L.ChMeas.CQI.DL.10',
            '[H] L.ChMeas.CQI.DL.11', '[H] L.ChMeas.CQI.DL.12',
            '[H] L.ChMeas.CQI.DL.13', '[H] L.ChMeas.CQI.DL.14',
            '[H] L.ChMeas.CQI.DL.15']


#%%
#reading Cell based PRB utilization ratios
# @st.cache
def prb(row): #adding PRB Util funciton
    try:
        return (100 * row['[H] L.ChMeas.PRB.DL.Used.Avg'] / row[
                '[H] L.ChMeas.PRB.DL.Avail'])

    except ZeroDivisionError:
        return 0

#%%
def scatter_plot(data_f, start_d, end_d):
    # a = alt.Chart(data_f[(data_f['DATETIME']>=start_d) & (data_f['DATETIME']<=end_d)]).mark_circle().encode(
    a = alt.Chart(data_f).mark_circle().encode(
        x="DATETIME", y='CQI Average',size='PRB Util %', color='TA Average',
        tooltip=['DATETIME', "CQI Average", 'PRB Util %', 'TA Average'])
    st.altair_chart(a, use_container_width=True)
    
#%%
def bar_plot():
    pass


#%%
#reading Cell based TA utilization ratios2
#@st.cache
def _ta_average(row):
    try:
        return ((row['[H] L.RA.TA.UE.Index0'] * 39) +
                (row['[H] L.RA.TA.UE.Index1'] * 195) +
                (row['[H] L.RA.TA.UE.Index2'] * 429) +
                (row['[H] L.RA.TA.UE.Index3'] * 819) +
                (row['[H] L.RA.TA.UE.Index4'] * 1521) +
                (row['[H] L.RA.TA.UE.Index5'] * 2769) +
                (row['[H] L.RA.TA.UE.Index6'] * 5109) +
                (row['[H] L.RA.TA.UE.Index7'] * 9516) +
                (row['[H] L.RA.TA.UE.Index8'] * 22269) +
                (row['[H] L.RA.TA.UE.Index9'] * 41769) +
                (row['[H] L.RA.TA.UE.Index10'] * 65169) +
                (row['[H] L.RA.TA.UE.Index11'] * 93600)
                ) / (row['[H] L.RA.TA.UE.Index0'] +
                row['[H] L.RA.TA.UE.Index1'] +
                row['[H] L.RA.TA.UE.Index2'] +
                row['[H] L.RA.TA.UE.Index3'] +
                row['[H] L.RA.TA.UE.Index4'] +
                row['[H] L.RA.TA.UE.Index5'] +
                row['[H] L.RA.TA.UE.Index6'] +
                row['[H] L.RA.TA.UE.Index7'] +
                row['[H] L.RA.TA.UE.Index8'] +
                row['[H] L.RA.TA.UE.Index9'] +
                row['[H] L.RA.TA.UE.Index10'] +
                row['[H] L.RA.TA.UE.Index11'])
    except ZeroDivisionError:
        return 0

#%%
#@st.cache
def _freq(x):
    if x == 'L':
        return 800
    elif x == 'T':
        return 1800
    elif x == 'E':
        return 2600
    else:
        return 0
#%%
@st.cache(allow_output_mutation=True)
def load_data1():
    data1 = pd.read_excel('/Users/macbookpro/Documents/python_exercises/TT_DATA/CL14/v2/SINR_CQI_20201201000000-20210106000000.xlsx',
                           usecols=cqi_cols)
    return data1

cqi_df = load_data1()
cqi_df['CQI Average'] = cqi_df.apply(_cqi_average, axis=1)

@st.cache(allow_output_mutation=True)
def load_data2():
    data2 = pd.read_excel('/Users/macbookpro/Documents/python_exercises/TT_DATA/CL14/v2/PRB_20201201000000-20210106000000.xlsx')
    return data2

prb_df = load_data2()
prb_df['PRB Util %'] = prb_df.apply(prb, axis=1)

@st.cache(allow_output_mutation=True)
def load_data3():
    data3 = pd.read_excel("/Users/macbookpro/Documents/python_exercises/TT_DATA/CL14/v2/4G Traffic with sampling rates.xlsx",
                        usecols=['DATETIME', 'CELL', 
                                 'Total PDCP Volume (MBytes)',
                                 '%ofDL16QAM', '%ofDL256QAM', '%ofDL64QAM',
                                 '%ofDLQPSK', '%ofUL16QAM', '%ofUL64QAM',
                                 '%ofULQPSK', 'DL PDCP Volume Mbytes',
                                 'UL PDCP Volume Mbytes'])
    return data3
pdcp_df = load_data3()

@st.cache(allow_output_mutation=True)
def load_data4():
    data4 = pd.read_excel('/Users/macbookpro/Documents/python_exercises/TT_DATA/CL14/v2/TA_20201201000000-20210106000000.xlsx')#, usecols=col_to_use)
    return data4

ta_df = load_data4()
ta_df['TA Average'] = ta_df.apply(_ta_average, axis=1)

@st.cache(allow_output_mutation=True)
def load_data5():    
    cqi_prb_df = pd.merge(cqi_df[['DATETIME', 'CELL', 'CQI Average']],
                          prb_df[['DATETIME', 'CELL', 'PRB Util %']],
                          on=['DATETIME', 'CELL'])
    cqi_prb_df['BAND'] = cqi_prb_df['CELL'].str[:1].apply(_freq)
    cqi_pdcp_df = pd.merge(cqi_df[['DATETIME', 'CELL', 'CQI Average']],
                           pdcp_df, on=['DATETIME', 'CELL'])
    cqi_pdcp_df['BAND'] = cqi_pdcp_df['CELL'].str[:1].apply(_freq)
    cqi_ta_df = pd.merge(cqi_df[['DATETIME', 'CELL', 'CQI Average']],
                         ta_df[['DATETIME', 'CELL', 'TA Average']],
                         on=['DATETIME', 'CELL'])
    cqi_ta_df['BAND'] = cqi_ta_df['CELL'].str[:1].apply(_freq)
    data5 = pd.merge(cqi_prb_df, cqi_ta_df[['DATETIME', 'CELL', 'TA Average']],
                     on=['DATETIME', 'CELL'])
    data5 = pd.merge(data5, pdcp_df, on=['DATETIME', 'CELL'])
    return data5

cqi_prb_TA_pdcp = load_data5()


#%%



logo = Image.open('/Users/macbookpro/Documents/Noya Sunumlar/3P_logoSREZ.png')

st.sidebar.image(logo, use_column_width=True)

start_d = st.sidebar.date_input('Start Date',
                                value=cqi_df['DATETIME'].min())
end_d = st.sidebar.date_input('End Date',
                              value=cqi_df['DATETIME'].max())
celllist = cqi_df['CELL'].unique().tolist()

cell_choice = st.sidebar.selectbox('CELL Name', celllist)

st.title('CQI Based Analysis for ' + cell_choice)

pdcp_analysis = st.sidebar.checkbox('PDCP Analysis')

@st.cache(suppress_st_warning=True)
def load_data6():
    data6 = cqi_prb_TA_pdcp.loc[(cqi_prb_TA_pdcp['CELL']==cell_choice)]
    data6 = data6.loc[(data6['DATETIME']>=pd.to_datetime(start_d)) &
                      (data6['DATETIME']<=pd.to_datetime(end_d))]
    return data6

df_prb_TA_pdcp = load_data6()

st.write('CQI Analysis and Visualiton with PRB & TA')
scatter_plot(df_prb_TA_pdcp, pd.to_datetime(start_d), pd.to_datetime(end_d))

if pdcp_analysis:
    st.write('Cell Based PDCP Visualiton')
    pdcp_type = st.sidebar.selectbox('Uplink - Downlik Selection', ['UL', 'DL'])
    
    if pdcp_type == 'UL':
        ul_pdcp = df_prb_TA_pdcp.iloc[:, [0,1,11,12,13,15]]
        ul_pdcp.set_index('DATETIME', inplace=True)
        ul_pdcp = ul_pdcp[['%ofUL16QAM', '%ofUL64QAM', '%ofULQPSK']].mul(ul_pdcp['UL PDCP Volume Mbytes'],axis=0)
        # st.bar_chart(ul_pdcp)
        # ax = ul_pdcp.plot.bar()
        # ax.xaxis.set_major_formatter(plt.FixedFormatter(ul_pdcp.index.to_series().dt.strftime("%Y-%m-%d")))
        # plt.show()
        ux = px.bar(ul_pdcp, x=ul_pdcp.index,
                    y=['%ofUL16QAM', '%ofUL64QAM', '%ofULQPSK'], barmode='group')
        st.plotly_chart(ux)
    else:
        dl_pdcp = df_prb_TA_pdcp.iloc[:, [0,1,7,8,9,10,14]]
        dl_pdcp.set_index('DATETIME', inplace=True)
        dl_pdcp = dl_pdcp[['%ofDL16QAM', '%ofDL64QAM', '%ofDL256QAM', '%ofDLQPSK']].mul(dl_pdcp['DL PDCP Volume Mbytes'],axis=0)
        # st.bar_chart(dl_pdcp)
        dx = px.bar(dl_pdcp, x=dl_pdcp.index,
                    y=['%ofDL16QAM', '%ofDL64QAM', '%ofDL256QAM', '%ofDLQPSK'], barmode='group')
        st.plotly_chart(dx)
        
        
#%%
# def file_selector(folder_path):
#     filenames = os.listdir(folder_path)
#     filenames.insert(0, "") # <-- default empty
#     selected_filename = st.sidebar.selectbox("Or select an example image", filenames)
#     return os.path.join(folder_path, selected_filename)

# image_selected = file_selector(folder_path="./images")

# if image_selected is not None and image_selected != './images\\': # <-- check for empty
#     image = np.array(Image.open(image_selected))
#     st.image(image)
#%%

# sns.barplot(x=ul_pdcp.index, y=['%ofUL16QAM', '%ofUL64QAM', '%ofULQPSK'],data=ul_pdcp.T)
# plt.xticks(rotation=270)
# plt.show()


