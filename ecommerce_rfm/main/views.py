from django.shortcuts import render
import joblib
import pandas as pd
from sklearn.cluster import KMeans
import random
import io
import base64
from matplotlib import pyplot as plt


loaded_rfm_model = joblib.load("ML_Model/rfm_model.joblib")
loaded_rfm_scores = joblib.load("ML_Model/rfm_scores.joblib")
loaded_rfm_sd = joblib.load("ML_Model/Scaled_data.joblib")

def index(request):
    if request.method == 'POST':
        k_value = int(request.POST.get('K'))
    else:
        k_value=5

    KMean_clust = KMeans(n_clusters= k_value, init= 'k-means++', max_iter= 10000)
    KMean_clust.fit(loaded_rfm_sd)

    #Find the clusters for the observation given in the dataset
    loaded_rfm_scores['Cluster'] = KMean_clust.labels_
    avg_mon=loaded_rfm_scores.groupby('Cluster')['M'].agg(['mean'])
    avg_rec=loaded_rfm_scores.groupby('Cluster')['R'].agg(['mean'])
    avg_freq=loaded_rfm_scores.groupby('Cluster')['F'].agg(['mean'])
    reframed_cluster=loaded_rfm_scores['Cluster'].apply(lambda x: x + 1)
    rfm_table = pd.DataFrame({'Cluster': sorted(reframed_cluster.unique()),
                                    'Ave. R': avg_rec['mean'].values,
                                    'Ave. F' : avg_freq['mean'].values,
                                    'Ave. M': avg_mon['mean'].values,})
    loaded_rfm_model['rfm_table']=rfm_table.to_html()

    plt.figure(figsize=(7,7))


    # Generate a list of random RGB values
    def generate_colors(num_clusters):
            colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_clusters)]
    # Convert RGB values to hex codes
            colors = ['#%02x%02x%02x' % c for c in colors]
            return colors

    Colors = generate_colors(k_value)
    loaded_rfm_scores['Color'] = loaded_rfm_scores['Cluster'].map(lambda p: Colors[p])
    ax = loaded_rfm_scores.plot(    
            kind="scatter", 
            x="Frequency", y="Recency",
            figsize=(10,8),
            c = loaded_rfm_scores['Color'],xlim=(0,60),ylim=(0,120))

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the PNG image in base64.
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    loaded_rfm_model['plot2']=graphic
    return render(request,'index.html',loaded_rfm_model)