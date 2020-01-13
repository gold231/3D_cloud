# -*- coding: utf-8 -*-
import pandas as pd
import pptk
from scipy.spatial import distance
import numpy as np


#This data have only the trees
tree_data = pd.read_csv('tree_data.csv')

#here we make the selection in z axis
tree_data = tree_data[tree_data['z'] > 1]

#here we make selection in the y axis
tree_data = tree_data[tree_data['y'] <= 4182508]

#Here we make filter by color
tree_data = tree_data[(tree_data['g'] > 0.20) & (tree_data['b'] < 0.20)]


#tree_data.to_csv('tree_data.csv',index=False,encoding='utf8')
v = pptk.viewer(tree_data[['x','y','z']].values,tree_data[['r','g','b']].values)




build_data = pd.read_csv('build_data.csv')
build_data = build_data[(build_data['z'] > 8) & (build_data['z'] < 12)]
build_data = build_data[(build_data['y'] >= 4182499) & (build_data['y'] <= 4182513)]
#build_data.to_csv('build_data.csv',index=False,encoding='utf8')


v = pptk.viewer(build_data[['x','y','z']].values,build_data[['r','g','b']].values)


#start from  x_min
y = build_data['x'].min()
graph = []

#Look now we use x axis
while y <= build_data['x'].max():
    
    tree_batch = tree_data[(tree_data['x'] >= y) & (tree_data['x'] < y + 0.2)]
    build_batch = build_data[(build_data['x'] >= y) & (build_data['x'] < y + 0.2)]
    
    y += 0.2
    
    dists = []
    if len(tree_batch) > 0 and len(build_batch) > 0:
        max_tree_y = tree_batch['y'].max()
        tree_batch = tree_batch[abs(tree_batch['y'] - max_tree_y) < 20 ]
        
        
        min_build_y = build_batch['y'].min()
        build_batch = build_batch[abs(build_batch['y'] - min_build_y)<3]

    
        for i in tree_batch[['x','y']].values:
            for j in build_batch[['x','y']].values:
                
                dists.append(distance.euclidean(i,j))
        
        graph.append(np.mean(dists))
    
