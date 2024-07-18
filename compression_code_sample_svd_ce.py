import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
fileName  = 'Frequency_Load_Outage.csv'
data_array = np.array(pd.read_csv(fileName))
pai = np.pi
evaluate_data =  np.zeros((data_array.shape[0],data_array.shape[1]))
for j in range(evaluate_data.shape[1]):
    for i in range(evaluate_data.shape[0]):
        evaluate_data[i][j] = data_array[i][j] + pai

unwrapped = np.zeros((evaluate_data.shape[0],evaluate_data.shape[1]))
for j in range(evaluate_data.shape[1]):
    for i in range(evaluate_data.shape[0]):
        fAng = evaluate_data[i][j];
        if(i==0):
            fUnwrap = 0;
            fPreAngle = evaluate_data[i][j]
            fAng = fAng + fUnwrap
        else:
            if(fPreAngle<pai/2 and fAng>3*pai/2):
                fUnwrap=fUnwrap-2*pai;
            elif(fPreAngle>3*pai/2 and fAng<pai/2):
                fUnwrap=fUnwrap+2*pai;
            fPreAngle = fAng;
            fAng = fAng + fUnwrap
        unwrapped[i][j] = fAng

aligned = np.zeros((unwrapped.shape[0],unwrapped.shape[1]))
for j in range(unwrapped.shape[1]):
    for i in range(unwrapped.shape[0]):
        #print(unwrapped[i][j]-unwrapped[0][j])
        aligned[i][j] = unwrapped[i][j]-unwrapped[0][j]

evaluate_data = aligned
from sklearn.preprocessing import normalize
n_data_diff = normalize(evaluate_data,axis=0,norm='l1')
n_data_diff_mean = np.mean(n_data_diff,axis=1)

diff_to_nominal = abs((n_data_diff.T-n_data_diff_mean).T)
entropy = -diff_to_nominal*np.log(1-diff_to_nominal)
weighted_entropy_list = np.max(entropy,axis=1)
window_size = 5

counter = 0
start = 0
end = 0
highEntropyPeriods = []
tolerance = 5
totalSamples = weighted_entropy_list.shape[0]
max_entropy = entropy.max()
for i in range(1,totalSamples):
    print(counter)
    if(weighted_entropy_list[i]>max_entropy/2):
        if(counter==0):
            start = i
        counter = tolerance
    else:
        if(counter>0):
            counter = counter - 1
            if(counter==0):
                end = i
                rangeList = []
                rangeList.append(start)
                rangeList.append(end)
                highEntropyPeriods.append(rangeList)

mergedPeriods = np.array(highEntropyPeriods)
for i in range(0,mergedPeriods.shape[0]):
    if(i==0 and mergedPeriods[i][0]==0):
        allPeriods = np.array([[0,mergedPeriods[i][1], np.median(entropy[0:mergedPeriods[i][0]-1]),entropy[0:mergedPeriods[i][0]-1].mean(),entropy[0:mergedPeriods[i][0]-1].max()]])
    elif(i==0 and mergedPeriods[i][0]>0):
        allPeriods = np.array([[0,mergedPeriods[i][0]-1, np.median(entropy[0:mergedPeriods[i][0]-1]),entropy[0:mergedPeriods[i][0]-1].mean(),entropy[0:mergedPeriods[i][0]-1].max()]])
    if(mergedPeriods[i][0]-allPeriods[allPeriods.shape[0]-1][1]>1):
        allPeriods = np.append(allPeriods, [[allPeriods[allPeriods.shape[0]-1][1]+1,mergedPeriods[i][0]-1,np.median(entropy[int(allPeriods[allPeriods.shape[0]-1][1]+1):int(mergedPeriods[i][0]-1)]),entropy[int(allPeriods[allPeriods.shape[0]-1][1]+1):int(mergedPeriods[i][0]-1)].mean(),entropy[int(allPeriods[allPeriods.shape[0]-1][1]+1):int(mergedPeriods[i][0]-1)].max()]], axis = 0)
    allPeriods = np.append(allPeriods, [[mergedPeriods[i][0],mergedPeriods[i][1],np.median(entropy[mergedPeriods[i][0]:mergedPeriods[i][1]]),entropy[mergedPeriods[i][0]:mergedPeriods[i][1]].mean(),entropy[mergedPeriods[i][0]:mergedPeriods[i][1]].max()]], axis = 0)
    if (i==mergedPeriods.shape[0]-1 and mergedPeriods[i][1]+1<entropy.shape[0]-1):
        allPeriods = np.append(allPeriods,[[mergedPeriods[i][1]+1,entropy.shape[0]-1,np.median(entropy[int(mergedPeriods[i][1]+1):int(entropy.shape[0]-1)]),entropy[int(mergedPeriods[i][1]+1):int(entropy.shape[0]-1)].mean(),entropy[int(mergedPeriods[i][1]+1):int(entropy.shape[0]-1)].max()]],axis = 0)

mean_diffs = []
max_diffs = []
min_diffs = []
tolerance_rate = 0.04

for i in range(0,allPeriods.shape[0]):
    this_chunk = n_data_diff[int(allPeriods[i][0]):int(allPeriods[i][1])+1]
    maximum_error_threshold = max(abs(np.max(this_chunk)*tolerance_rate),abs(np.min(this_chunk)*tolerance_rate))
    print(maximum_error_threshold)
    u,s,vh = np.linalg.svd(this_chunk, full_matrices = False)
    for reducedDimension in range(1,n_data_diff.shape[1]+1):
        new_u = u[:,0:int(reducedDimension)]
        new_s = s[0:int(reducedDimension)]
        new_vh = vh[0:int(reducedDimension),:]
        recon = np.dot(new_u*new_s,new_vh)
        if(abs(recon-this_chunk).max()<=maximum_error_threshold):
            break;
    print(reducedDimension)
    if(i==0):
        entropy_to_dimension=np.array([np.append(allPeriods[i][2:],reducedDimension)])
    else:
        entropy_to_dimension = np.append(entropy_to_dimension,[np.append(allPeriods[i][2:],reducedDimension)],axis=0)
     #start from 2 because 0 and 1 are periods start and end
    new_u = u[:,0:int(reducedDimension)]
    new_s = s[0:int(reducedDimension)]
    new_vh = vh[0:int(reducedDimension),:]
    recon = np.dot(new_u*new_s,new_vh)
    print("Reconstructed "+str(recon.shape[0])+" samples.")
    if(i==0):
        recon_full_new = recon
    else:
        recon_full_new = np.append(recon_full_new,recon,axis = 0)
    diff = abs(recon-this_chunk)
    [].append(diff.mean())
    max_diffs.append(diff.max())
    mean_diffs.append(diff.mean())
    min_diffs.append(diff.min())
    np.save(fileName+'_u'+str(i),new_u)
    np.save(fileName+'_s'+str(i),new_s)
    np.save(fileName+'_vh'+str(i),new_vh)

mean_diff = np.mean(mean_diffs)
max_diff = np.max(max_diffs)
avg_dimension = np.matmul(allPeriods[:,1]-allPeriods[:,0],entropy_to_dimension[:,3])/allPeriods[-1,1]
CR = n_data_diff.shape[1]/(avg_dimension+1)
print("mean_diff:"+str(mean_diff))
print("max_diff:"+str(max_diff))
print("CR:"+str(CR))
fig, axs = plt.subplots(1,1)
axs.plot(weighted_entropy_list)
plt.show()

#if(entropy_to_dimension.shape[0]>0):
#    entropy_to_dimension = entropy_to_dimension[1:] # exclude the first chunk because it will contains both event and ambient data
#
#np.save('entropy_to_dimension_training'+fileName,entropy_to_dimension)
#mean_diffs
#max_diffs
#min_diffs

#old_way


mean_diffs = []
max_diffs = []
min_diffs = []
for i in range(0,allPeriods.shape[0]):
    this_chunk = n_data_diff[int(allPeriods[i][0]):int(allPeriods[i][1])+1]
    u,s,vh = np.linalg.svd(this_chunk, full_matrices = False)
    reducedDimension = 10
    if(allPeriods[i][3]==-1):
        reducedDimension = n_data_diff.shape[1]/n_data_diff.shape[1]
    else:
        for n_scores in range(1,n_data_diff.shape[1]):
            print(n_scores)
            if(sum(s)==0.0):
                reducedDimension = 1
            elif(sum(s[0:n_scores])/sum(s)>0.96):
                print(sum(s[0:n_scores])/sum(s))
                reducedDimension = n_scores
                break;
    print(reducedDimension)
    if(i==0):
        entropy_to_dimension=np.array([np.append(allPeriods[i][2:],reducedDimension)])
    else:
        entropy_to_dimension = np.append(entropy_to_dimension,[np.append(allPeriods[i][2:],reducedDimension)],axis=0)
    new_u = u[:,0:int(reducedDimension)]
    new_s = s[0:int(reducedDimension)]
    new_vh = vh[0:int(reducedDimension),:]
    recon = np.dot(new_u*new_s,new_vh)
    print("Reconstructed "+str(recon.shape[0])+" samples.")
    if(i==0):
        recon_full = recon
    else:
        recon_full = np.append(recon_full,recon,axis = 0)
    diff = abs(recon-this_chunk)
    [].append(diff.mean())
    max_diffs.append(diff.max())
    mean_diffs.append(diff.mean())
    min_diffs.append(diff.min())
    np.save(fileName+'_u'+str(i),new_u)
    np.save(fileName+'_s'+str(i),new_s)
    np.save(fileName+'_vh'+str(i),new_vh)

mean_diff = np.mean(mean_diffs)
max_diff = np.max(max_diffs)
avg_dimension = np.matmul(allPeriods[:,1]-allPeriods[:,0],entropy_to_dimension[:,3])/allPeriods[-1,1]
CR = n_data_diff.shape[1]/(avg_dimension+1)
print("mean_diff:"+str(mean_diff))
print("max_diff:"+str(max_diff))
print("CR:"+str(CR))