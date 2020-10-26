#Importing the libraries
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import linear_model
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

plt.style.use('seaborn-bright')

# Draw a title and some text to the app:
'''
# Market Size Forecasting  Demo
#### Using Multiple Linear Regression and Marco Indicators from IMF.


Select a county and a market to view the historic and forecast data.

'''

datadf = pd.read_csv("data.csv")
marketdatadf = pd.read_csv("marketdata.csv")

col1, col2 = st.beta_columns(2)

countries = col1.selectbox(
    "Select a country", list(["China", "India", "United Kingdom", "United States"])
    )

market = col2.selectbox(
    "Select a Market", list(["Construction", "Healthcare Services", "Chemicals", "Machinery Manufacturing"])
    )


marketdf = marketdatadf[marketdatadf['Country'] == countries]
marketdf = marketdf[['Year', 'Country', market]]
marketdf = marketdf[marketdf['Year']<=2019]
mar_val19 = marketdf.iloc[8,2]
datadf = datadf[datadf['Country'] == countries]
datadf = datadf[['Year', 'Country', 'GDP', 'Inflation','Volume of Imports of goods' , 'Volume of exports of goods', market]]
traindf = datadf[datadf['Year']<=2019]
predictdf = datadf[datadf['Year']>2019]
predict_updf = predictdf
predict_v1 = predictdf
Y = traindf[market]
X = traindf[['GDP', 'Inflation','Volume of Imports of goods' , 'Volume of exports of goods']]

#traindf

#Y
#X

#predictdf
Z = predictdf[['GDP', 'Inflation','Volume of Imports of goods' , 'Volume of exports of goods']]

reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(X,Y)
#reg.coef_
GDPcoef = reg.coef_[0]
Inflationcoef = reg.coef_[1]
VIGcoef = reg.coef_[2] 
VEGcoef = reg.coef_[3]

#GDPcoef
#Inflationcoef
#VIGcoef
#VEGcoef

predictdf[market] = GDPcoef * predictdf['GDP'] + Inflationcoef * predictdf['Inflation'] +  VIGcoef * predictdf['Volume of Imports of goods'] + VEGcoef * predictdf['Volume of exports of goods']
model_YOY = predictdf[['Year', market]]
#st.write('this is predictdf')
#predictdf

#predict_v1[market] = reg.predict(predict_v1[['GDP', 'Inflation','Volume of Imports of goods' , 'Volume of exports of goods']])
#predict_v1[market] = reg.predict(Z)
#st.write('this is predictv1')

#predict_v1


modelled_value =  pd.DataFrame( columns = ['Year', 'Market_Value']) 
modelled_value =  pd.DataFrame({'Year': [2019, 2020,2021,2022,2023,2024,2025], 'Market_Value': mar_val19}) 
for i in range (1,7):
    (modelled_value.loc[i, 'Market_Value']) = (1+ model_YOY.iloc[i-1,1])*modelled_value.iloc[i-1,1]

modelled_value = modelled_value.rename(columns = {'Market_Value' : 'Forecast Market Value'})
comb_value1 = modelled_value
modelled_value = modelled_value.T
new_header = modelled_value.iloc[0]
modelled_value = modelled_value[1:]
modelled_value.columns = new_header
modelled_value = modelled_value.drop([2019], axis = 1)


fulldf = traindf.append(predictdf)
#fulldf

'''
### Figure 1: Historic Market Size 2011 - 2019, $ Billon
'''

fig, ax = plt.subplots(figsize=(12,6))
ax.set_title("Historic Market Size 2011 - 2019, $ Billon", fontsize=14)
sns.barplot(x = marketdf['Year'], y = marketdf[market], color = '#1B8E86', ci = None)
ax.set_xlabel("Year",fontsize=12)
ax.set_ylabel("Market Value ($ Billion)", fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_visible(False)
st.pyplot(fig)

st.write('The chart above shows the historic market size (hard data) for selected market and country')

st.write('We are creating a model using YOY growth rates of the historic market size, GDP, Inflation, volume of goods imported, and volume of goods exported.')
st.write('We are using a multiple linear regression model that fits the input data into the following equation to predict the forecast YOY growth of the market')
st.latex('''
        Y(Market YOY Growthrate) = w1*(GDP YOY) + w2*(Inflation YOY) + w3*(Volume of goods imported YOY)+ w4*(Volume of goods exported YOY)
        
''')

st.write('''
        Where, 
        w1 = GDP weightage arrived from the model
''')

st.write('''
        w2 = Inflation weightage arrived from the model
''')

st.write('''
        w3 = Volume of goods imported weightage arrived from the model
''')

st.write('''
        w4 = Volume of goods exported weightage arrived from the model
''')

'''
### Figure 2: Forecast Market YOY Growth Rate vs Macro Indicators
'''

fig = plt.figure(figsize= (12,8))
ax1 = plt.subplot(111)
ax1.set_title("Forecast YOY Growth Rate, Market(prediced) vs Macro Indicators", fontsize=14)
ax1.plot(predictdf['Year'],predictdf['GDP'], 'o-b')
ax1.plot(predictdf['Year'],predictdf['Inflation'], 'o-g')
ax1.plot(predictdf['Year'],predictdf['Volume of Imports of goods'], 'o-m')
ax1.plot(predictdf['Year'],predictdf['Volume of exports of goods'], 'o-k')
ax1.plot(predictdf['Year'],predictdf[market], 'o--r')
ax1.legend(['GDP', 'Inflation', 'Volume of Imports of goods', 'Volume of exports of goods', market], bbox_to_anchor = (1.0,-0.05), frameon = False, ncol = 5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
st.pyplot(fig)

'''
### Table 1: Forecast Market YOY Growth Rate vs Macro Indicators
'''


tabledf = predictdf.drop(['Country'], axis = 1)
tabledf = tabledf.T
new_header = tabledf.iloc[0]
tabledf = tabledf[1:]
tabledf.columns = new_header
st.table(tabledf)

st.write('The chart and table above shows the forecasted YOY for the selected market and country. The respective forecasts of the macro indicators that were used as an input for prediction are also shown here.')

'''
### Table 2: Forecast Market Value, 2019 - 2025, $ Billion
'''
st.table(modelled_value)

st.write('The table above shows the forecasted market size in $ Billion.')

'''
### Table 3: Default weightages arrived for each marco indicator from the regression model 
'''

tempdata = {'Indicator': ['GDP', 'Inflation', 'Volume of Imports of goods', 'Volume of exports of goods'], 'Weightage':[round(GDPcoef,3),round(Inflationcoef,3), round(VIGcoef,3),round(VEGcoef,3)]}
coefdf = pd.DataFrame(tempdata)
coefdf = coefdf.T
new_header = coefdf.iloc[0]
coefdf = coefdf[1:]
coefdf.columns = new_header
st.table(coefdf)

st.write('The above table shows the default weightages for each macro indicator from the regression model. These weightages can be tweaked to adjust the forecasts manually')

st.write('Enter weightages below and click on update forecast button to view the new forecasts based on your input')
col3, col4, col5, col6 = st.beta_columns(4)

#GDPcoef_updated = col3.number_input('GDP')
#Inflationcoef_updated = col4.number_input('Inflation')
#VIGcoef_updated = col5.number_input('Volume of Imports of goods')
#VEGcoef_updated = col6.number_input('Volume of Exports of goods')


GDPcoef_updated = (col3.number_input('GDP'))
Inflationcoef_updated = (col4.number_input('Inflation'))
VIGcoef_updated = (col5.number_input('Volume of Imports of goods'))
VEGcoef_updated = (col6.number_input('Volume of Exports of goods'))


submit = st.button('Update Forecast')
if submit:    
    predict_updf[market] = GDPcoef_updated * predict_updf['GDP'] + Inflationcoef_updated * predict_updf['Inflation'] +  VIGcoef_updated * predict_updf['Volume of Imports of goods'] + VEGcoef_updated * predict_updf['Volume of exports of goods']
    upmodel_YOY = predict_updf[['Year', market]]
    upmodelled_value =  pd.DataFrame( columns = ['Year', 'Market_Value']) 
    upmodelled_value =  pd.DataFrame({'Year': [2019, 2020,2021,2022,2023,2024,2025], 'Market_Value': mar_val19}) 
    for i in range (1,7):
        (upmodelled_value.loc[i, 'Market_Value']) = (1+ upmodel_YOY.iloc[i-1,1])*upmodelled_value.iloc[i-1,1]
    
    upmodelled_value = upmodelled_value.rename(columns = {'Market_Value' : ' Updated_Forecast_Market_Value'})
    comb_value2 = upmodelled_value
    upmodelled_value = upmodelled_value.T
    new_header = upmodelled_value.iloc[0]
    upmodelled_value = upmodelled_value[1:]
    upmodelled_value.columns = new_header
    upmodelled_value = upmodelled_value.drop([2019], axis = 1)


    '''
    ### Figure 3: Updated Forecast Market YOY Growth Rates vs Macro Indicators
    '''
    
    fig = plt.figure(figsize= (12,8))
    ax1 = plt.subplot(111)
    #ax1.set_title("Forecast YOY Growth Rate", fontsize=18)
    ax1.plot(predict_updf['Year'],predict_updf['GDP'], 'o-b')
    ax1.plot(predict_updf['Year'],predict_updf['Inflation'], 'o-g')
    ax1.plot(predict_updf['Year'],predict_updf['Volume of Imports of goods'], 'o-m')
    ax1.plot(predict_updf['Year'],predict_updf['Volume of exports of goods'], 'o-k')
    ax1.plot(predict_updf['Year'],predict_updf[market], 'o--r')
    ax1.legend(['GDP', 'Inflation', 'Volume of Imports of goods', 'Volume of exports of goods', market], bbox_to_anchor = (1.0,-0.05), frameon = False, ncol = 5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    st.pyplot(fig)

    '''
    ### Table 4: Updated Forecast Market YOY Growth Rates vs Macro Indicators
    '''

    predict_updf = predict_updf.drop(['Country'], axis = 1)
    predict_updf = predict_updf.T
    new_header = predict_updf.iloc[0]
    predict_updf = predict_updf[1:]
    predict_updf.columns = new_header
    
    st.table(predict_updf)

    st.write('The chart and table above shows the updated forecasted YOY based on the updated weightages for the selected market and country.')

    '''
    ### Table 5: Updated Forecast Market Value, 2019 - 2025, $ Billion
    '''
    st.table(upmodelled_value)

    st.write('The table above shows the updated forecast market value based on the new weightages')


    comb_value = pd.concat([comb_value1, comb_value2], axis=1, sort=False)
    comb_value = comb_value.T
    new_header = comb_value.iloc[0]
    comb_value = comb_value[1:]
    comb_value.columns = new_header
    comb_value = comb_value.drop(comb_value.index[1])
    comb_value = comb_value.drop([2019], axis = 1)

    '''
    ### Table 6: Original Vs Updated Forecast Market Value, 2019 - 2025, $ Billion
    '''
    st.table(comb_value)

    st.write('The table above shows the original and the updated forecast market value')




