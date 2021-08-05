#This script is taking two sets of rainbow measures (red sweep and blue sweep) and compares all the cells in order to find the best wavelength cut for each pair of cells. 
from tkinter.constants import N
from matplotlib import colors
#import Small_Functions as sf
import numpy as np
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio

#First of all we load all the data. For that, we ask to the user where are the measure files of the blue cell and the red cell, 
#read each rainbow measure for each measured cell and save it on an array (BlueCell_megaarray, RecCell_megaarray) and its corresponding _names in a list.

 
BlueCell_File_Directory = os.path.join(filedialog.askdirectory(title='Select the directory of the BLUE CELL', initialdir=os.getcwd(), mustexist=True),'TXT files')   # Ask for the directory where all the EQE and JV curve files are                 #Just adding the TXT file at the end of the path because there is an extra folder to open
BlueCell_Available_folders = os.listdir(BlueCell_File_Directory)
BlueCells_names = []                                                                    #initialize the list of names   
for folder in BlueCell_Available_folders:               
    if not folder.endswith(".txt"):                                                     #find for all the folders inside the specified path whic corresponds to the measured cells without reading the calibration files (which are .txt files)
        
        cell_pathlist = os.listdir(os.path.join(BlueCell_File_Directory,folder))        #enter to each foler corresponding to each measured cell
        for file in cell_pathlist:                                                      
            if file.endswith(".txt"):                                                   #And store the data in the corresponding megaarray        
                load_txt_filepath = os.path.join(BlueCell_File_Directory,folder,file)
                actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
                BlueCells_names.append(folder)
                try: 
                    BlueCell_megaarray
                except NameError: 
                    BlueCell_megaarray = actual_cell_data
                else: 
                    BlueCell_megaarray = np.dstack((BlueCell_megaarray, actual_cell_data))
                
#This part is doing the same as before, but for the red cell.               
RedCell_File_Directory = os.path.join(filedialog.askdirectory(title='Select the directory of the RED CELL', initialdir=os.getcwd(), mustexist=True), 'TXT files')  # Ask for the directory where all the EQE and JV curve files are
RedCell_Available_folders = os.listdir(RedCell_File_Directory)
RedCells_names = []
for folder in RedCell_Available_folders:
    if not folder.endswith(".txt"):
        
        cell_pathlist = os.listdir(os.path.join(RedCell_File_Directory,folder))
        for file in cell_pathlist:
            if file.endswith(".txt"):
                load_txt_filepath = os.path.join(RedCell_File_Directory,folder,file)
                actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
                RedCells_names.append(folder)
                try: 
                    RedCell_megaarray
                except NameError: 
                    RedCell_megaarray = actual_cell_data
                else: 
                    RedCell_megaarray = np.dstack((RedCell_megaarray, actual_cell_data))

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


#Now we will calculate the IOBC (improvement over best cell) for each pair of cells and then select the bests combinations
# IOBC = []                                                               #First start the list that will have all the IOBC values for each ocmbination
# for n in range (0, len(Summ_megaarray[0,0,:])):                         #Loop arround all the possible combinations
#     [i,j] = Summs_indexes[n,:]                                          
#     best_cell_PCE = np.amax((RedCell_megaarray[:,5,j], BlueCell_megaarray[:,5,i]))  #And find the maxim efficiency of blue and red cell
#     IOBC_actual = 100*((np.amax(Summ_megaarray[:,5,n])/best_cell_PCE)-1)                 #Afterwards, calculate the IOBC as the division of the maximum of the summ divided by the best cell PCE and rest 1
#     IOBC.append(IOBC_actual)                                                        #Finally put the actual IOBC in the IOBC list


filenames = []
gs = gridspec.GridSpec(1, 1)
for i in range (0,1):
    for j in range (len(RedCell_megaarray[0,0,:])):             #Now we plot all the results for the N best cells combinations
        i=8
        nn = np.intersect1d(np.argwhere(Summs_indexes[:,0] == i), np.argwhere(Summs_indexes[:,1] == j))
        xred = RedCell_megaarray[:,0,j]
        yred = RedCell_megaarray[:,5,j]
        xblue = BlueCell_megaarray[:,0,i]
        yblue = BlueCell_megaarray[:,5,i]
        xtot = Summ_megaarray[:,0,nn[0]]
        ytot = Summ_megaarray[:,5,nn[0]]
        IoBC = 100*(np.amax(ytot)/(np.amax((yred, yblue)))-1)

        fig = plt.figure(figsize=(8, 5))
        fig.suptitle('Full rainbow analysis for ' + BlueCells_names[i] +'Blue cell / ' + RedCells_names[j] + 'Red cell', fontsize=16)
        ax1 = fig.add_subplot(gs[0, 0])  
        ax1.plot(xred, yred, color='tab:red', label = RedCells_names[j] + 'Red Cell')
        ax1.plot(xblue, yblue, color='tab:blue', label = BlueCells_names[i] + 'Blue Cell' )
        ax1.plot(xtot, ytot, color='tab:green', label = 'Red cell + Blue Cell')
        ax1.set_title('Efficiency vs Dividing Wavelength. IOBC(%)='+ str(IoBC))
        ax1.legend(loc='upper right')
        ax1.set_ylim(0,5)
        ax1.set(xlabel='Dividing Wavelength /nm', ylabel='Efficiency /%')
        filename = (str(j)+'.png')
        plt.savefig(filename)
        filenames.append(filename)



    # Build GIF
    with imageio.get_writer(BlueCells_names[i]+'.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)
