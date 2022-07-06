##########################################################################################
# Autor: dr Sreejita Ghosh
# The following has been modified from our MATLAB implemntation.  The following is based on Chawla et al's SMOTE (https://arxiv.org/pdf/1106.1813.pdf) , but modified to be able to handle missing values implicitly, without the need for imputation with the sklearn's imblearn smote needs currently as of 06/07/2022. This geodesicSMOTE is explained in our paper here https://arxiv.org/pdf/1106.1813.pdf
#
# After git cloning add the path to the nan_smote repository in your code. Currently the choice of dissimilarity are just two: Euclidean and cosine dissimilarity. 
# Example: 
# from nan_smote import smote_all_class
# overs_ztrainX, overs_trainY=smote_all_class(ztrainX,y_train,'Euclid')      
#
# The code has been written purely from the functional perspective. Anyone interested in improving upon the implementation is highly encouraged to join this repo. 
##########################################################################################
##########################################################################################

import numpy as np
import pandas as pd
import os
import sys

def nearestNeighbor(original_vector, dissType):
    #This function computes dissimilarity/ distance matrix based on Euclidean distance with partial distance
    # strategy, or based on Cosine dissimilarity, as selected by the user. The arguments to be passed are the minority class dataframe which needs to be amplified, and the disstance or dissiilarity the user needs.
    #It returns the dissimialrity/ distance matrix.
    x=original_vector.copy()  
    x.fillna(0, inplace=True) 
    sqX=lambda x:np.nansum(x**2,axis=1)    
# the * operator on two Numpy matrices is equivalent to the .* operator in Matlab    
    if dissType=='Euclid':
        nNull=sqX(x).reshape((len(sqX(x)),1))
        nData=sqX(original_vector).reshape((len(sqX(original_vector)),1))           
        prod=x.dot(x.T)
        D_all=x.shape[1]   
        D_obs=D_all-original_vector.isnull().sum(axis=1)
        #computing Euclidean distance and normalizing based on observed dimensions
        d_euclid=(D_obs/D_all)*np.round(nNull +(np.transpose(nNull-2*prod)),5)  
        dist_mat=d_euclid
        
    else: 
        # if angle-based dissimilarity    
        # computing the unit vectors 
        denom=sqX(x).reshape((len(sqX(x)),1))        
        nNull=x/denom 
        nData=original_vector/sqX(original_vector).reshape((len(sqX(original_vector)),1)) 
        thetas = np.zeros((np.shape(original_vector)[0],np.shape(original_vector)[0]))        
        for idx in range(np.shape(nNull)[0]):
            for jdx in range(np.shape(nNull)[0]):
                if idx==jdx:
                    thetas[idx,jdx] = 0
                else:
                    dotproduct=np.dot(nNull.iloc[idx,:],nNull.iloc[jdx,:])                   
                    thetas[idx,jdx] = np.arccos(dotproduct)                            
        dist_mat=thetas        
    return dist_mat

def nan_smote_per_class(original_dataset,smote_percent, k, dissType):
    # This function uses the distance/ dissimilarity matrix from nearestNeighbor() and finds the k nearest neighbour of a randomly selected sample, and then based on whether the selected dissimialarity is Cosine based or Euclidean, it creates synthetic samples on the manifold (for cosine) or on a plane, 'somewhere' (this random effect is added to prevent always having a synthetic sample created on the midpoint of two real samples) on the path between the randomly selected sample and one of its k-nearest neighbours.
    a=0
    a0=0
    N=np.shape(original_dataset)[0]*(smote_percent/100)         
    perm_indices=np.random.permutation(np.shape(original_dataset)[0]-1)    
    if dissType == 'Euclid' :
        # EUCLIDEAN DISTANCE    
        dist_mat=nearestNeighbor(original_dataset,dissType)            
        while a<N:  
            if a0>len(perm_indices)-1:
    # if the #iterations exceeds length of the minority class rows then counters and per_indices are reset
                a0=0
                perm_indices=np.random.permutation(np.shape(original_dataset)[0]-1)
            ind_selected=perm_indices[a0]
            selected_sample=original_dataset.iloc[ind_selected,:]
            # only distances wrt the selected sample index in d_interest
            d_interest=dist_mat.iloc[ind_selected,:] 
            #sorting out the distances based on selected_sample
            nn_indices=np.argsort(d_interest)
            sortedDistance=np.sort(d_interest)              
            neighbor_samples=original_dataset.iloc[nn_indices,:] # K nearest neighbors of sample of interest
            differ=neighbor_samples-original_dataset.iloc[ind_selected,:]# differences between the 5 
            useable_indices=np.where((np.sum(differ.isnull(),axis=1)<0.93*np.shape(differ)[1]))[0]
            if len(useable_indices)<2: # since there is one entry (the first one in useable_indices) which is the distance of the ind_selected from itself      
                print(useable_indices)                    
            kNearestIdxs=useable_indices[1:k+1]
            randOutKNN = np.random.permutation(kNearestIdxs)[0]
            nn_dist=sortedDistance[useable_indices[1:k+1]]            
            alpha_nn=np.random.rand(1,len(selected_sample))* np.reshape(original_dataset.iloc[randOutKNN,:].values, (1, len(selected_sample)))
            selected_sample=selected_sample.values
            selected_sample=selected_sample.reshape(1,len(selected_sample))
            # adding random vector alpha_nn to selected sample to have the s'=s+alpha effect
            synthetic=np.nansum([selected_sample,alpha_nn],axis=0) 
            if a==0:
                synthetic_sample=synthetic
            else:
                synthetic_sample=np.concatenate((synthetic_sample,synthetic),axis=0)                
            a=a+1
            a0=a0+1
    else: 
        # COSINE DISSIMILARITY
        thetas= nearestNeighbor(original_dataset, dissType)        
        # smaller angles mean near neighbors
        nearAngles=np.sort(thetas, axis=1)
        nearIdx  =np.argsort(thetas, axis=1)                
        while a<N:                          
            if a0>len(perm_indices)-1:
    # if the #iterations exceeds length of the minority class rows then counters and per_indices are reset
                a0=0
                perm_indices=np.random.permutation(np.shape(original_dataset)[0]-1)
            ind_selected=perm_indices[a0]            
            kNearestIdxs = nearIdx[ind_selected, np.where(nearAngles[ind_selected,:]>0)[0][0]:np.where(nearAngles[ind_selected,:]>0)[0][k-1]]                                   
            randOutKNN = np.random.permutation(kNearestIdxs)[0]
            m=original_dataset.iloc[ind_selected,:]            
            x=thetas[ind_selected, randOutKNN].T/np.sin(thetas[ind_selected, randOutKNN]).T * original_dataset.iloc[randOutKNN,:]- m.T*np.cos(thetas[ind_selected,randOutKNN])        
            test=   np.random.rand(1,np.shape(x)[0])[0]*(x-m)            
            normtest= np.sqrt(np.nansum(test**2))
            theta2=normtest.copy()            
            test2M= m*np.cos(theta2) + np.sin(theta2)/theta2*test                        
            temp=test2M/np.sqrt(np.nansum(test2M**2))
            synthetic=np.reshape(temp.values, (1,len(test2M)))                        
            if a==0:
                synthetic_sample=synthetic
            else:             
                synthetic_sample=np.concatenate((synthetic_sample,synthetic),axis=0)                
            a=a+1
            a0=a0+1
           
    if np.not_equal(np.shape(synthetic_sample)[1],np.shape(original_dataset)[1]):
        synthetic_sample=np.transpose(synthetic_sample)
    new_dataset=np.concatenate((original_dataset, synthetic_sample), axis=0)
    return new_dataset

def smote_all_class(all_class_data, all_class_labs, dissType): 
    # gets the synthetic_samples from nan_smote_per_class() in a loop until the number of toal samples in minority class (real + synthstic) is equivalent to number of samples in majority class.
    # It accepts the original dataframe (majority and minority class together) and all teh labels (majority and minority class together), and the user-defined choice of 'Euclid' or 'Cosine' for computation of distance matrix and thereby the synthetic samples
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
                synthetic=nan_smote_per_class(classData,oversample_percent,5, dissType)
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
            synthetic=nan_smote_per_class(classData,oversample_percent,5, dissType)
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

