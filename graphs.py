
# coding: utf-8

# In[15]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode, plot
init_notebook_mode(connected=True)

import random

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# # graphs

# In[2]:

df_dup = pd.read_csv("vSocial_Dup.csv")
df_dup.head()


# In[3]:

df_norm = pd.read_csv("vSocial_Norm.csv")
df_norm.head()


# In[4]:

df_tamper = pd.read_csv("vSocial_Tamper.csv")
df_tamper.head()


# In[5]:

df_drop = pd.read_csv("vSocial_Drop.csv")
df_drop.head()


# In[6]:

df_norm


# In[7]:

plt.figure(figsize=(10,6))
plt.scatter(list(df_dup.Time)[:100], list(df_dup["No."])[:100], s=1)
plt.xlabel("Time")
plt.ylabel("# Packets")
plt.show()


# In[8]:

plt.figure(figsize=(10,6))
plt.scatter(list(df_norm.Time), list(df_norm["No."]), s=1)
plt.xlabel("Time")
plt.ylabel("# Packets")
plt.show()


# In[9]:

data = [go.Scatter(
          x=list(df_dup.Time),
          y=list(df_dup['No.']))]

layout = dict(
    title='Time series with range slider and selectors',
    xaxis=dict(
        rangeslider=dict(
            visible = True
        )
    )
)

iplot(dict(data=data, layout=layout))


# In[10]:

data = [go.Scatter(
          x=list(df_norm.Time),
          y=list(df_norm['No.']))]

layout = dict(
    title='Time Series of Packages Sent',
    xaxis=dict(
        rangeslider=dict(
            visible = True
        )
    )
)

iplot(dict(data=data, layout=layout))


# In[11]:

def getPacketSpeeds(df):
    speedArray = []
    for i in range(0, 60):
        speedArray.append(len(df[(df.Time >= i) & (df.Time <= i+1)]))
    return speedArray


# In[12]:

plt.figure(figsize=(10,6))
plt.plot(list(range(0,60)), getPacketSpeeds(df_dup), '-o')
plt.xlabel("Time")
plt.ylabel("# Packets/second")
plt.show()


# In[13]:

plt.figure(figsize=(10,6))
plt.plot(list(range(0,60)), getPacketSpeeds(df_norm), '-o')
plt.xlabel("Time")
plt.ylabel("# Packets/second")
plt.show()


# In[129]:

def randomColor():
    (r,g,b) = [str(random.randint(1,255)), str(random.randint(1,255)), str(random.randint(1,255))]
    color = 'rgb(' + r + ','+ g + ',' + b + ')'
    return color

def updateVisibility(selected_type):
    visibilityValues = []
    for plot_datum in plot_data:
        if plot_datum['Type'] == selected_type:
            visibilityValues.append(True)
        else:
            visibilityValues.append(False)
    return visibilityValues

norm_total = {'x': list(df_norm.Time[df_norm.Time <= 60][0::50]), 'y': list(df_norm['No.'][0::50])}
dup_total = {'x': list(df_dup.Time[df_dup.Time <= 60][0::50]), 'y': list(df_dup['No.'][0::50])}
tamper_total = {'x': list(df_tamper.Time[df_tamper.Time <= 60][0::50]), 'y': list(df_tamper['No.'][0::50])}
drop_total = {'x': list(df_drop.Time[df_drop.Time <= 60][0::50]), 'y': list(df_drop['No.'][0::50])}

norm_speed = {'x': list(range(0,60)), 'y': getPacketSpeeds(df_norm)}
dup_speed = {'x': list(range(0,60)), 'y': getPacketSpeeds(df_dup)}
tamper_speed = {'x': list(range(0,60)), 'y': getPacketSpeeds(df_tamper)}
drop_speed = {'x': list(range(0,60)), 'y': getPacketSpeeds(df_drop)}

data = []
buttons_data = []
buttons_labels = ["Total Packets Sent", "Packet Speed"]
plot_data = [{"Title": "No Manipulation", "Type": "Total Packets Sent", "Data": norm_total, "Color": "rgb(77, 82, 91)"}, 
             {"Title": "Duplication", "Type": "Total Packets Sent", "Data": dup_total, "Color": "rgb(104, 156, 249)"}, 
             {"Title": "Tampering", "Type": "Total Packets Sent", "Data": tamper_total, "Color": "rgb(249, 104, 104)"}, 
             {"Title": "Dropping", "Type": "Total Packets Sent", "Data": drop_total, "Color": "rgb(104, 249, 133)"},
             {"Title": "No Manipulation", "Type": "Packet Speed", "Data": norm_speed, "Color": "rgb(77, 82, 91)"}, 
             {"Title": "Duplication", "Type": "Packet Speed", "Data": dup_speed, "Color": "rgb(104, 156, 249)"}, 
             {"Title": "Tampering", "Type": "Packet Speed", "Data": tamper_speed, "Color": "rgb(249, 104, 104)"}, 
             {"Title": "Dropping", "Type": "Packet Speed", "Data": drop_speed, "Color": "rgb(104, 249, 133)"}]

for button_label in buttons_labels:
    buttons_data.append(dict(
        label = button_label,
        method = 'update',
        args = [{'visible': updateVisibility(button_label)}]
    ))
    
for plot_datum in plot_data:
    data.append(go.Scatter(
        x=plot_datum["Data"]["x"],
        y=plot_datum["Data"]["y"],
        mode='lines+markers',
        line=dict(
            color=plot_datum["Color"],
            width=1
        ),
        marker = dict(
            size = 1
        ),
        name=plot_datum["Title"],
        text=plot_datum["Title"],
        visible=(plot_datum["Type"]=='Total Packets Sent')
    ))
    

updatemenus = list([
    dict(active=0,
         buttons= buttons_data,
         direction = 'down',
         pad = {'r': 10, 't': 10},
         showactive = True,
         x = 0,
         xanchor = 'center',
         y = 1.3,
         yanchor = 'top'
    )
])
    
layout = dict(
    title='Packet Monitoring',
    updatemenus = updatemenus,
    xaxis=dict(
        rangeslider=dict(),
        autorange=True
    ),
    annotations=[
        go.layout.Annotation(
            x=0.5004254919715793,
            y=-0.16191064079952971,
            showarrow=False,
            text='Time (sec)',
            xref='paper',
            yref='paper',
            font=dict(
                size=16,
            ),
        ),
#         go.layout.Annotation(
#             x=-0.06944728761514841,
#             y=0.4714285714285711,
#             showarrow=False,
#             text='Packets',
#             textangle=-90,
#             xref='paper',
#             yref='paper'
#         )
    ],
)

fig = dict(data=data, layout=layout)
iplot(fig)


# In[50]:

# plot(fig, filename='chart.html')


# # tsfresh example

# In[ ]:

download_robot_execution_failures()


# In[41]:

df, y = load_robot_execution_failures()
df = df[['id','time', 'F_x']]
df.head()


# In[60]:

y.head()


# In[73]:

y


# In[66]:

df.time.unique()


# In[44]:

df[df.id == 3][['time', 'F_x']].plot(x='time', title='Success example (id 3)', figsize=(12, 6));
df[df.id == 20][['time', 'F_x']].plot(x='time', title='Failure example (id 20)', figsize=(12, 6));


# In[45]:

extraction_settings = ComprehensiveFCParameters()


# In[46]:

X = extract_features(df, 
                     column_id='id', column_sort='time',
                     default_fc_parameters=extraction_settings,
                     impute_function= impute)


# In[47]:

X.head()


# In[48]:

X_filtered = extract_relevant_features(df, y, 
                                       column_id='id', column_sort='time', 
                                       default_fc_parameters=extraction_settings)


# In[49]:

X_filtered.head()


# In[50]:

X_train, X_test, X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X, X_filtered, y, test_size=.4)


# In[51]:

cl = DecisionTreeClassifier()
cl.fit(X_train, y_train)
print(classification_report(y_test, cl.predict(X_test)))


# In[52]:

cl.n_features_


# In[53]:

cl2 = DecisionTreeClassifier()
cl2.fit(X_filtered_train, y_train)
print(classification_report(y_test, cl2.predict(X_filtered_test)))


# In[54]:

cl2.n_features_


# # classifier

# In[88]:

dataframes = [{'data':df_norm, 'y_value':False}, {'data':df_tamper, 'y_value':True}, {'data':df_dup, 'y_value':True}]
y_store = []
df_store = []
sample_id = 0
for df_i in range(len(dataframes)):
    y_val = dataframes[df_i]['y_value']
    for chunk in np.split(np.array(getPacketSpeeds(dataframes[df_i]['data'])), 4):
        y_store.append(y_val)
        for chunk_i in range(len(chunk)):
            df_store.append({'id':sample_id, 'time':chunk_i, 'val':chunk[chunk_i]})
        sample_id += 1

df = pd.DataFrame(df_store)
y = pd.Series(y_store)


# In[106]:

df


# In[90]:

y


# In[91]:

extraction_settings = ComprehensiveFCParameters()


# In[92]:

X = extract_features(df, 
                     column_id='id', column_sort='time',
                     default_fc_parameters=extraction_settings,
                     impute_function= impute)


# In[93]:

X.head()


# In[105]:

X_filtered = extract_relevant_features(df, y, 
                                       column_id='id', column_sort='time',
                                       default_fc_parameters=extraction_settings)


# In[102]:

X_filtered.head()


# In[97]:

X_train, X_test, X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X, X_filtered, y, test_size=.4)


# In[127]:

cl = DecisionTreeClassifier()
cl.fit(X_train, y_train)
print(classification_report(y_test, cl.predict(X_test)))


# In[128]:

cl.n_features_


# In[100]:

cl2 = DecisionTreeClassifier()
cl2.fit(X_filtered_train, y_train)
print(classification_report(y_test, cl2.predict(X_filtered_test)))


# In[ ]:



