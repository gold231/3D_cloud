# -*- coding: utf-8 -*-
import pandas as pd
import pptk
from scipy.spatial import distance
import numpy as np


#This data have only the trees
tree_data = pd.read_csv('tree_data.csv')

#here we make the selection in z axis
tree_data = tree_data[tree_data['z'] > 1]

#here we make selection in the x axis
tree_data = tree_data[tree_data['x'] >= 561301]

#Here we make filter by color
tree_data = tree_data[(tree_data['g'] > 0.20) & (tree_data['b'] < 0.20)]


#tree_data.to_csv('tree_data.csv',index=False,encoding='utf8')
v = pptk.viewer(tree_data[['x','y','z']].values,tree_data[['r','g','b']].values)




build_data = pd.read_csv('build_data.csv')
build_data = build_data[(build_data['z'] > 8) & (build_data['z'] < 12)]
build_data = build_data[(build_data['x'] >= 561298) & (build_data['x'] <= 561330)]
#build_data.to_csv('build_data.csv',index=False,encoding='utf8')


#v = pptk.viewer(build_data[['x','y','z']].values,build_data[['r','g','b']].values)


y = build_data['y'].min()
graph = []

while y <= build_data['y'].max():
    
    tree_batch = tree_data[(tree_data['y'] >= y) & (tree_data['y'] < y + 0.2)]
    build_batch = build_data[(build_data['y'] >= y) & (build_data['y'] < y + 0.2)]
    
    y += 0.2
    
    dists = []
    if len(tree_batch) > 0 and len(build_batch) > 0:
        min_tree_x = tree_batch['x'].min()
        tree_batch = tree_batch[abs(tree_batch['x'] - min_tree_x) < 20 ]
        
        
        max_x = build_batch['x'].max()
        build_batch = build_batch[abs(build_batch['x'] - max_x)<3]

    
        for i in tree_batch[['x','y']].values:
            for j in build_batch[['x','y']].values:
                
                dists.append(distance.euclidean(i,j))
        
        graph.append(np.mean(dists))
    



#plt.plot(graph)
#plt.savefig('graph_south.png')











