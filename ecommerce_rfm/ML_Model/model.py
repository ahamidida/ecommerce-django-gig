import io
import base64

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import joblib



df=pd.read_csv("data.csv")

df=df.drop(df[df['total_purchase'] < 0].index)
df=df[df['total_purchase']<= 761000.0]

df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.weekday
stats = df.groupby('weekday')['total_purchase'].agg(['mean', 'std'])
weekday_table = pd.DataFrame({'Weekday': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                               'Mean': stats['mean'].values,
                               'Standard Deviation ': stats['std'].values,})

df['day_type'] = df['weekday'].apply(lambda x: 'Weekend' if x in [3, 4] else 'Weekday')
weekday_data = df.loc[df['day_type'] == 'Weekday', 'total_purchase']
weekend_data = df.loc[df['day_type'] == 'Weekend', 'total_purchase']
plt.hist(weekday_data, bins=20, alpha=0.5, color='blue', label='WorkingDays')
plt.hist(weekend_data, bins=20, alpha=0.5, color='red', label='Weekends')
plt.title('Daily Demand Distribution by Day Type')
plt.xlabel('Total Purchase')
plt.ylabel('Frequency')
plt.legend()

buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)

# Encode the PNG image in base64.
image_png = buffer.getvalue()
buffer.close()
graphic = base64.b64encode(image_png).decode('utf-8')

latest_date = df['date'].max()

rfm_scores = df.groupby('user_id').agg({
            'date' : lambda x: (latest_date - x.max()).days,
            'order_id': lambda x: len(x),
            'total_purchase':lambda x: x.sum()})

rfm_scores['date'] = rfm_scores['date'].astype(int)

rfm_scores.rename(columns={'date':'Recency',
                                'order_id':'Frequency',
                                'total_purchase':'Monetary'},inplace=True)
quantiles = rfm_scores.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()
def RScoring(x,p,d):
            if x <= d[p][0.25]:
                return 1
            elif x <= d[p][0.50]:
                return 2
            elif x <= d[p][0.75]: 
                return 3
            else:
                return 4

def FnMScoring(x,p,d):
            if x <= d[p][0.25]:
                return 4
            elif x <= d[p][0.50]:
                return 3
            elif x <= d[p][0.75]: 
                return 2
            else:
                return 1

def handle_neg_n_zero(num):
            if num <= 0:
                return 1
            else:
                return num
rfm_scores['R'] = rfm_scores['Recency'].apply(RScoring, args=('Recency',quantiles,))
rfm_scores['F'] = rfm_scores['Frequency'].apply(FnMScoring, args=('Frequency',quantiles,))
rfm_scores['M'] = rfm_scores['Monetary'].apply(FnMScoring, args=('Monetary',quantiles,))
#Apply handle_neg_n_zero function to Recency and Monetary columns 
rfm_scores['Recency'] = [handle_neg_n_zero(x) for x in rfm_scores.Recency]
rfm_scores['Monetary'] = [handle_neg_n_zero(x) for x in rfm_scores.Monetary]
Log_Tfd_Data = rfm_scores[['Recency', 'Frequency', 'Monetary']].apply(np.log10, axis = 1).round(3)
from sklearn.preprocessing import StandardScaler

#Bring the data on same scale
scaleobj = StandardScaler()
Scaled_Data = scaleobj.fit_transform(Log_Tfd_Data)

#Transform it back to dataframe
Scaled_Data = pd.DataFrame(Scaled_Data, index = rfm_scores.index, columns = Log_Tfd_Data.columns)
context={'table':weekday_table.to_html(),
         'plot1':graphic,}

joblib.dump(context,'rfm_model.joblib')
joblib.dump(rfm_scores,'rfm_scores.joblib')
joblib.dump(Scaled_Data,'Scaled_data.joblib')
