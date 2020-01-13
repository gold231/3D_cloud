# -*- coding: utf-8 -*-
from laspy.file import File
import numpy as np
import pandas as pd
import pptk




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
#data = data[data['c'] == 2]
data = data[(data['y'] >= 4182471) & (data['y'] <= 4182595)]
data = data[(data['x'] >= 561103) & (data['x'] <= 561362)]


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

