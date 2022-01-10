import numpy as np
import pandas as pd
import os

def nearestNeighbor(original_vector,ind_selected,k):
    x=original_vector.copy()
    x.fillna(0)
    sqX=lambda x:np.nansum(x**2,axis=1)
    sumsquared=sqX(original_vector).reshape((len(sqX(original_vector)),1))
# the * operator on two Numpy matrices is equivalent to the .* operator in Matlab
    prod=x.dot(x.T)
    D_all=x.shape[1]
   # D_obs=D_all-original_vector.iloc[ind_selected,:].isnull().sum()
    D_obs=D_all-original_vector.isnull().sum(axis=1)
    
#computing Euclidean distance and normalizing based on observed dimensions
    d_euclid=(D_obs/D_all)*np.round(sumsquared +(np.transpose(sumsquared-2*prod)),5)   
    d_interest=d_euclid.iloc[ind_selected,:]
#sorting out the distances 
    sortedIndex=np.argsort(d_interest)
    sortedDistance=np.sort(d_interest)
    nn_indices=sortedIndex
    neighbor_samples=original_vector.iloc[nn_indices,:] # 5 nearest neighbors of selected_sample are found
    differ=neighbor_samples-original_vector.iloc[ind_selected,:]# differences between the 5 nearest neighbors and selected_sample computed
    useable_indices=np.where((np.sum(differ.isnull(),axis=1)<0.93*np.shape(differ)[1]))[0]
    if len(useable_indices)<2: # since there is one entry (the first one in useable_indices) which is the distance of the ind_selected from itself      
        print(useable_indices)
        #     useable_indices checks if any of the diff is only NaNs-such indices in diff should be rejected right away
    idx=useable_indices[1:k+1] # starting from index 1 instead of 0 to exclude the sample ind_selected
    nn_distances=sortedDistance[idx]
    return idx, nn_distances

def nan_smote_per_class(original_dataset,smote_percent,k):
    a=0
    N=np.shape(original_dataset)[0]*(smote_percent/100)     
    while a<N:  
        index=np.random.permutation(np.shape(original_dataset)[0]-1)[:2]# selects indices that are not 1 or the index of the last sample
        if np.sum(np.in1d(index,0))>0:
            index=index[~np.in1d(index,0)]
        index=index[0]        
        selected_sample=original_dataset.iloc[index,:]
    #ensures that the selected_sample itself isn't one of the selected nearest neighbors
        nn_indices, nn_dist=nearestNeighbor(original_dataset,index,5)
        neighbor_samples=original_dataset.iloc[nn_indices,:] # K nearest neighbors of selected_sample are found
       # differ=neighbor_samples-selected_sample # differences between the K nearest neighbors and selected_sample computed
        idx=np.random.permutation(len(nn_indices))[0]
        alpha_nn=np.random.rand(1,len(selected_sample))*nn_dist[idx]
        selected_sample=selected_sample.values
        selected_sample=selected_sample.reshape(1,len(selected_sample))
        synthetic=np.nansum([selected_sample,alpha_nn],axis=0)
        if a==0:
            synthetic_sample=synthetic
        else:
            synthetic_sample=np.concatenate((synthetic_sample,synthetic),axis=0)                
        a=a+1
    if np.not_equal(np.shape(synthetic_sample)[1],np.shape(original_dataset)[1]):
        synthetic_sample=np.transpose(synthetic_sample)
    new_dataset=np.concatenate((original_dataset, synthetic_sample), axis=0)
    return new_dataset

def smote_all_class(all_class_data, all_class_labs):
    #all_class_labs=all_class_labs.as_array()
    unique_class=np.unique(all_class_labs)
    class_counts=np.bincount(all_class_labs)
    class_counts=class_counts[unique_class]
    ind_max=np.argmax(class_counts)   
    newSamples=np.array([])
    newLabels=np.array([])
    if len(unique_class)>2:
        minorities=np.delete(unique_class,unique_class[ind_max])
        for c in minorities:
            classData = all_class_data.loc[all_class_labs==c,:]
            while np.sum(all_class_labs==unique_class[ind_max])-np.shape(classData)[0]>10:
                oversample_percent=round(class_counts[ind_max]/class_counts[c])*100
                synthetic=nan_smote_per_class(classData,oversample_percent,3)
                if class_counts[ind_max] < np.shape(np.concatenate((classData,synthetic), axis=0))[0]:
                    C=classData
                    deficit=np.sum(all_class_labs==unique_class[ind_max])-np.shape(classData)[0]
                    C=np.concatenate((C,synthetic[:deficit,:]),axis=0)
                    classData=C
                    if len(newSamples)==0:
                        newSamples=synthetic[:deficit,:]
                    else:
                        newSamples=np.concatenate((newSamples,synthetic[:deficit,:]),axis=0)                   
                    newLabels=np.append(newLabels,np.ones((np.shape(synthetic[:deficit,:])[0],1))*c) 
                else:
                    if len(newSamples)==0:
                        newSamples=synthetic
                    else:
                        newSamples=np.concatenate((newSamples,synthetic),axis=0)
                    classData=pd.concat([classData, synthetic])
                    newLabels=np.append(newLabels,np.ones((np.shape(synthetic)[0],1))*c)
                    
    else:
        unique_class=unique_class[class_counts>0]
        class_counts=class_counts[class_counts>0]
        c=unique_class[np.argmin(class_counts)]
        classData = all_class_data.loc[all_class_labs==c,:]
        while np.sum(all_class_labs==ind_max)-np.shape(classData)[0]>10:
            oversample_percent=round(class_counts[ind_max]/class_counts[c])*100
            synthetic=nan_smote_per_class(classData,oversample_percent,3)
            if class_counts[ind_max] < np.shape(np.concatenate((classData,synthetic), axis=0))[0]:
                C=classData
                deficit=np.sum(all_class_labs==unique_class[ind_max])-np.shape(classData)[0]
                C=np.concatenate((C,synthetic[:deficit,:]),axis=0)
                classData=C
                if len(newSamples)==0:
                    newSamples=synthetic[:deficit,:]
                else:
                    newSamples=np.concatenate((newSamples,synthetic[:deficit,:]),axis=0)                   
                newLabels=np.append(newLabels,np.ones((np.shape(synthetic[:deficit,:])[0],1))*c) 
            else:
                if len(newSamples)==0:
                    newSamples=synthetic
                else:
                    newSamples=np.concatenate((newSamples,synthetic),axis=0)
                classData=pd.concat([classData, synthetic])
                newLabels=np.append(newLabels,np.ones((np.shape(synthetic)[0],1))*c)
        
    newSamples=pd.DataFrame(newSamples)
    old_names=newSamples.columns
    new_names=all_class_data.columns
    newSamples.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    newTrainX=pd.concat([all_class_data,newSamples])
    newTrainLabs=np.concatenate((all_class_labs,newLabels),axis=0).astype('int16')   
    return newTrainX, newTrainLabs                                                         