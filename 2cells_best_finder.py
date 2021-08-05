#This script is taking two sets of rainbow measures (red sweep and blue sweep) and compares all the cells in order to find the best wavelength cut for each pair of cells. 
from tkinter.constants import N
from matplotlib import colors
#import Small_Functions as sf
import numpy as np
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
################################################
#First of all we load all the data. For that, we ask to the user where are the measure files of the blue cell and the red cell, 
#read each rainbow measure for each measured cell and save it on an array (BlueCell_megaarray, RecCell_megaarray) and its corresponding _names in a list.


BlueCell_Files_Directory =  filedialog.askdirectory(title='Select the directory of the BLUE CELL', initialdir=os.getcwd(), mustexist=True)
BlueCell_Files_Directory_rainbow= os.path.join(BlueCell_Files_Directory,'TXT files')   #Define the directory where all the rainbow sweeps are saved
BlueCell_Files_Directory_EQE = os.path.join(BlueCell_Files_Directory, 'EQE')           #Define the directory where all the EQE data is saved
BlueCell_Available_folders = os.listdir(BlueCell_Files_Directory_rainbow)
BlueCells_names = []                                                                    #initialize the list of names   
for folder in BlueCell_Available_folders:               
    if not folder.endswith(".txt"):                                                     #find for all the folders inside the specified path whic corresponds to the measured cells without reading the calibration files (which are .txt files)
        cell_pathlist = os.listdir(os.path.join(BlueCell_Files_Directory_rainbow,folder))        #enter to each foler corresponding to each measured cell
        for file in cell_pathlist:                                                      
            if file.endswith(".txt"):                                                   #And store the data in the corresponding megaarray        
                load_txt_filepath = os.path.join(BlueCell_Files_Directory_rainbow,folder,file)
                actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
                BlueCells_names.append(folder)
                try: 
                    BlueCell_megaarray
                except NameError: 
                    BlueCell_megaarray = actual_cell_data
                else: 
                    BlueCell_megaarray = np.dstack((BlueCell_megaarray, actual_cell_data))

for file in BlueCell_Files_Directory_EQE:
    for ii in range(len(BlueCells_names)):
    	if file.endswith(BlueCells_names[ii]+'.txt'):
            load_txt_filepath = os.path.join(BlueCell_Files_Directory_EQE,file)
            actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
            try: 
                BlueCell_EQE
            except NameError: 
                BlueCell_EQE = actual_cell_data
            else: 
                BlueCell_EQE = np.dstack((BlueCell_EQE, actual_cell_data))


#This part is doing the same as before, but for the red cell. 
RedCell_Files_Directory = filedialog.askdirectory(title='Select the directory of the RED CELL', initialdir=os.getcwd(), mustexist=True)
RedCell_Files_Directory_rainbow = os.path.join(RedCell_Files_Directory, 'TXT files')  # Ask for the directory where all the EQE and JV curve files are
RedCell_Files_Directory_EQE = os.path.join(RedCell_Files_Directory, 'EQE')
RedCell_Available_folders = os.listdir(RedCell_Files_Directory_rainbow)
RedCells_names = []
for folder in RedCell_Available_folders:
    if not folder.endswith(".txt"):
        
        cell_pathlist = os.listdir(os.path.join(RedCell_Files_Directory_rainbow,folder))
        for file in cell_pathlist:
            if file.endswith(".txt"):
                load_txt_filepath = os.path.join(RedCell_Files_Directory_rainbow,folder,file)
                actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
                RedCells_names.append(folder)
                try: 
                    RedCell_megaarray
                except NameError: 
                    RedCell_megaarray = actual_cell_data
                else: 
                    RedCell_megaarray = np.dstack((RedCell_megaarray, actual_cell_data))

for file in RedCell_Files_Directory_EQE:
    for ii in range(len(RedCells_names)):
    	if file.endswith(RedCells_names[ii]+'.txt'):
            load_txt_filepath = os.path.join(RedCell_Files_Directory_EQE,file)
            actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
            try: 
                RedCell_EQE
            except NameError: 
                RedCell_EQE = actual_cell_data
            else: 
                RedCell_EQE = np.dstack((RedCell_EQE, actual_cell_data))

#Here we make the arrays corresponding to all the possible sums between cells. 
Sums_names = []                                                                     #first start the list of cell names for each sum
for i in range(len(BlueCell_megaarray[0,0,:])):                                  #Loop for the blue cell           
    for j in range (len(RedCell_megaarray[0,0,:])):                              #Loop for the red cell
        Summ_actual_array = np.zeros(RedCell_megaarray[:,:,j].shape)                            #first initialize the numpy array of the actual summ

        for r_wl_ind in range (len(RedCell_megaarray[:,0,j])):                                        #This small part is added just in case that red and blue sweep does not have the same cutting wavelenght order (it seareches for the corresponding cutting wavelenght of the red array to the blue array and makes the summ)
            b_wl_ind = np.where(BlueCell_megaarray[:,0,i]==RedCell_megaarray[r_wl_ind,0,j])           #Find the index of the blue cell array that has the same dividing wavelength as the red cell array
            Summ_actual_array [r_wl_ind,:] = BlueCell_megaarray[b_wl_ind,:,i] + RedCell_megaarray[r_wl_ind,:,j]    #For that wavelength, sum of all the parameters    
            Summ_actual_array [r_wl_ind,0:1] = Summ_actual_array [r_wl_ind,0:1]/2                     #Except the dividing wavelength (which should be the same for all the data) and sun power (which will be similar, so we make the average)
        
        Summ_actual_name = BlueCells_names[i] +'Blue cell / ' + RedCells_names[j] + 'Red cell'  #Saving the name of the corresponding cell pairs
        Summ_actual_index = np.array([i, j])                                        #and the index of the BlueCell and RedCell megaarrays so afterwards se can plot them
        
        if i==0 and j==0:                                                             #Store the data in new arrays for the first iteration
            Summ_megaarray = Summ_actual_array
            Sums_names.append(Summ_actual_name)
            Summs_indexes = Summ_actual_index
           
        else:
            Summ_megaarray = np.dstack((Summ_megaarray, Summ_actual_array))            #and stacking them for the nexts iterations
            Sums_names.append(Summ_actual_name)
            Summs_indexes = np.vstack((Summs_indexes, Summ_actual_index))


#Now we will calculate the IOBC (improvement over best cell) for each pair of cells and the maximum PCE that each combination achieves
IoBC = []                                                               #First start the list that will have all the IOBC values for each combination
maxPCE = []                                                            #And the PCE_max values
for n in range (0, len(Summ_megaarray[0,0,:])):                         #Loop arround all the possible combinations
    [i,j] = Summs_indexes[n,:]                                          
    best_cell_PCE = np.amax((RedCell_megaarray[:,5,j], BlueCell_megaarray[:,5,i]))  #And find the maxim efficiency of blue and red cell
    IoBC_actual = 100*((np.amax(Summ_megaarray[:,5,n])/best_cell_PCE)-1)                 #Afterwards, calculate the IOBC as the division of the maximum of the summ divided by the best cell PCE and rest 1
    IoBC.append(IoBC_actual)                                                        #Finally put the actual IOBC in the IOBC list
    maxPCE.append(np.amax(Summ_megaarray[:,5,n]))


#We add a IoBC treshold, since tanking the best and the worts cell will give something with a really high maxPCE but a IoBC near to 0
# for t in range(len(IoBC)):
#     if IoBC[t] <= 8:
#         IoBC[t] = 0
#         maxPCE[t] = 0


#Now find the N combinations with higher IOBC, and appart the N with higher total PCE
N = 15                              
IoBC_max = np.argsort(np.array(IoBC))#[-N:]            #Find the corresponding indexes in the sum matrixes corresponding to N maximum IoBC
maxPCE_max = np.argsort(np.array(maxPCE))#[-N:]        #Find the corresponding indexes in the sum matrixes corresponding to N maximum maxPCE
#IoBC_max_and_maxPCE_max = np.intersect1d(IoBC_max, maxPCE_max) #Find the indexes that are to both cases
IoBC_max_and_maxPCE_max = np.array([IoBC,maxPCE], dtype=[('IoBC_max', 'f4'),('maxPCE','f4')])
IoBC_max_and_maxPCE_max = np.argsort(IoBC_max_and_maxPCE_max, order=('maxPCE', 'IoBC_max'))



gs = gridspec.GridSpec(1, 1)
for k in range (N):             #Now we plot all the results for the N best cells combinations
    n = IoBC_max[-k-1]
    [i,j] = Summs_indexes[n,:]
    xred = RedCell_megaarray[:,0,j]
    yred = RedCell_megaarray[:,5,j]
    xblue = BlueCell_megaarray[:,0,i]
    yblue = BlueCell_megaarray[:,5,i]
    xtot = Summ_megaarray[:,0,n]
    ytot = Summ_megaarray[:,5,n]
    summ_max_index = np.amax(ytot)
    fig = plt.figure(figsize=(8, 5))
    fig.suptitle('Full rainbow analysis for ' + Sums_names[n], fontsize=16)
    ax1 = fig.add_subplot(gs[0, 0])  
    ax1.plot(xred, yred, color='tab:red', label = RedCells_names[j] + 'Red Cell')
    ax1.plot(xblue, yblue, color='tab:blue', label = BlueCells_names[i] + 'Blue Cell' )
    ax1.plot(xtot, ytot, color='tab:green', label = 'Red cell + Blue Cell')
    ax1.set_title('Efficiency vs Dividing Wavelength. IoBC(%)='+ str(IoBC[n]))
    ax1.legend(loc='upper right')
    ax1.set(xlabel='Dividing Wavelength /nm', ylabel='Efficiency /%')
    filename = ('Best combination_'+str(k)+'.png')
    plt.savefig(filename)
