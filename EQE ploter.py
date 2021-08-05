from tkinter.constants import N
from matplotlib import colors
import numpy as np
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plab
from scipy.interpolate import make_interp_spline


# BlueCell_Files_Directory =  filedialog.askdirectory(title='Select the directory of the BLUE CELL', initialdir=os.getcwd(), mustexist=True)
# BlueCell_Files_Directory_rainbow= os.path.join(BlueCell_Files_Directory,'TXT files')   #Define the directory where all the rainbow sweeps are saved
# BlueCell_Files_Directory_EQE = os.path.join(BlueCell_Files_Directory, 'EQE')           #Define the directory where all the EQE data is saved
# BlueCell_Available_folders = os.listdir(BlueCell_Files_Directory_rainbow)
# BlueCells_names = []                                                                    #initialize the list of names   
# for folder in BlueCell_Available_folders:               
#     if not folder.endswith(".txt"):                                                     #find for all the folders inside the specified path whic corresponds to the measured cells without reading the calibration files (which are .txt files)
#         cell_pathlist = os.listdir(os.path.join(BlueCell_Files_Directory_rainbow,folder))        #enter to each foler corresponding to each measured cell
#         for file in cell_pathlist:                                                      
#             if file.endswith(".txt"):                                                   #And store the data in the corresponding megaarray        
#                 load_txt_filepath = os.path.join(BlueCell_Files_Directory_rainbow,folder,file)
#                 actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
#                 BlueCells_names.append(folder)
#                 try: 
#                     BlueCell_megaarray
#                 except NameError: 
#                     BlueCell_megaarray = actual_cell_data
#                 else: 
#                     BlueCell_megaarray = np.dstack((BlueCell_megaarray, actual_cell_data))

# for file in BlueCell_Files_Directory_EQE:
#     for ii in range(len(BlueCells_names)):
#     	if file.endswith(BlueCells_names[ii]+'.txt'):
#             load_txt_filepath = os.path.join(BlueCell_Files_Directory_EQE,file)
#             actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
#             try: 
#                 BlueCell_EQE
#             except NameError: 
#                 BlueCell_EQE = actual_cell_data
#             else: 
#                 BlueCell_EQE = np.dstack((BlueCell_EQE, actual_cell_data))


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


RedCell_Files_Directory_EQE_files = os.listdir(RedCell_Files_Directory_EQE)
for file in RedCell_Files_Directory_EQE_files:    
    for ii in range(len(RedCells_names)):
        filename = str(RedCells_names[ii]+'.txt')
        print(filename)
        if file.endswith(filename):
            load_txt_filepath = os.path.join(RedCell_Files_Directory_EQE,file)
            actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)            

            try: 
                RedCell_EQE
            except NameError: 
                RedCell_EQE = actual_cell_data
            else: 
                RedCell_EQE = np.dstack((RedCell_EQE, actual_cell_data))

print(RedCell_EQE)
### PLOTING
gs = gridspec.GridSpec(1, 1)
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(gs[0, 0])
fig.suptitle('OSC prameters vs DivW vs thickness for P3HT:O-IDFBR (0712OF1) - Right side', fontsize=16)
colors_ = plab.cm.jet(np.linspace(0,1,len(RedCell_EQE[0,0,:])))
for i in range(len(RedCell_EQE[0,0,:])):
    x=RedCell_EQE[:,0,i]
    y=RedCell_EQE[:,1,i]
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(x.min(), 900, 500)
    Y_ = X_Y_Spline(X_)
    ax1.plot(X_,Y_, label = RedCells_names[i], color=colors_[i])

ax1.legend(loc='upper right')
plt.show()