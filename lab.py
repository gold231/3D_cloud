# -*- coding: utf-8 -*-
from laspy.file import File
import numpy as np
import pptk
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import distance





inFile = File('Test_group1_densified_point_cloud.las', mode='r')
color = np.vstack([inFile.Red/65280.,inFile.Green/65280.,inFile.Blue/65280.]).transpose()

#inFile.visualize(mode="elevation")
coor_data = np.vstack([inFile.x, inFile.y, inFile.z]).transpose()


data = pd.DataFrame(coor_data,columns=['x','y','z'])
data['c'] = inFile.Classification
data['r'] = color[:,0]
data['g'] = color[:,1]
data['b'] = color[:,2]



data['zn'] = data['z']/data['z'].max()
data['yn'] = data['y']/data['y'].max()
data['xn'] = data['x']/data['x'].max()
data = data[(data['z'] < 20) & (data['z'] > 1)]
data = data[data['c'] == 2]
#data = data[(data['y'] >= 4182471) & (data['y'] <= 4182595)]
#data = data[(data['x'] >= 561103) & (data['x'] <= 561362)]


#kmeans = KMeans(n_clusters=2)
#kmeans.fit(data[['r','g','b']].values)
#
#data['clusters'] = kmeans.predict(data[['r','g','b']].values)
#data = data[data['z'] > 2]
#
#c_data = data[data['c'] == 6]
#
#
v = pptk.viewer(data[['x','y','z']].values,data[['r','g','b']].values)


#This data have only the trees
tree_data = pd.read_csv('tree_data.csv')
tree_data = tree_data[tree_data['z'] > 8]
#tree_data = tree_data[tree_data['x'] >= 561300]
#tree_data.to_csv('tree_data.csv',index=False,encoding='utf8')
v = pptk.viewer(tree_data[['x','y','z']].values,tree_data[['r','g','b']].values)


build_data = pd.read_csv('build_data.csv')
build_data = build_data[(build_data['z'] > 8) & (build_data['z'] < 12)]
build_data = build_data[build_data['x'] >= 561300]
#build_data.to_csv('build_data.csv',index=False,encoding='utf8')
v = pptk.viewer(build_data[['x','y','z']].values,build_data[['r','g','b']].values)


a = 0
y = build_data['y'].min()
graph = []

while y <= build_data['y'].max():
    
    tree_batch = tree_data[(tree_data['y'] >= y) & (tree_data['y'] < y + 0.2)]
    build_batch = build_data[(build_data['y'] >= y) & (build_data['y'] < y + 0.2)]
    
    y += 0.2
    
    dists = []
    if len(tree_batch) > 0 and len(build_batch) > 0:
    
        for i in tree_batch[['x','y']].values:
            for j in build_batch[['x','y']].values:
                dists.append(distance.euclidean(i,j))
        
        graph.append(np.mean(dists))
    
    
    















