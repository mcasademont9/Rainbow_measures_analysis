from tkinter.constants import N
from matplotlib import colors
import numpy as np
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plab

File_Directory = os.path.join(filedialog.askdirectory(title='Select the directory of the measures', initialdir=os.getcwd(), mustexist=True),'TXT files')   # Ask for the directory where all the EQE and JV curve files are                 #Just adding the TXT file at the end of the path because there is an extra folder to open
Available_folders = os.listdir(File_Directory)
Cells_names = []                                                                    #initialize the list of names   
for folder in Available_folders:               
    if not folder.endswith(".txt"):                                                     #find for all the folders inside the specified path whic corresponds to the measured cells without reading the calibration files (which are .txt files)
        
        cell_pathlist = os.listdir(os.path.join(File_Directory,folder))        #enter to each foler corresponding to each measured cell
        for file in cell_pathlist:                                                      
            if file.endswith(".txt"):                                                   #And store the data in the corresponding megaarray        
                load_txt_filepath = os.path.join(File_Directory,folder,file)
                actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
                Cells_names.append(folder)
                try: 
                    Cell_megaarray
                except NameError: 
                    Cell_megaarray = actual_cell_data
                else: 
                    Cell_megaarray = np.dstack((Cell_megaarray, actual_cell_data))


gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
fig.suptitle('OSC prameters vs DivW vs thickness for P3HT:O-IDFBR (0712OF1) - Right side', fontsize=16)
colors_ = plab.cm.jet(np.linspace(0,1,len(Cell_megaarray[0,0,:])))
for i in range(len(Cell_megaarray[0,0,:])):
    ax1.plot(Cell_megaarray[:,0,i], Cell_megaarray[:,5,i], label = Cells_names[i], color=colors_[i])
    ax2.plot(Cell_megaarray[:,0,i], Cell_megaarray[:,4,i], label = Cells_names[i], color=colors_[i])
    ax3.plot(Cell_megaarray[:,0,i], Cell_megaarray[:,3,i], label = Cells_names[i], color=colors_[i])
    ax4.plot(Cell_megaarray[:,0,i], Cell_megaarray[:,2,i], label = Cells_names[i], color=colors_[i])
    #ax1.set_title('Efficiency vs Dividing Wavelength.')

#ax1.set_title('PCE vs Dividing wavelenth')
ax1.legend(loc='upper left')
ax1.set(xlabel='Dividing Wavelength [nm]', ylabel='Efficiency [%]')

#ax2.set_title('FF vs Dividing wavelenth')
#ax2.legend(loc='lower left')
ax2.set(xlabel='Dividing Wavelength [nm]', ylabel='FF [%]')

#ax3.set_title('FF vs Dividing wavelenth')
#ax3.legend(loc='lower left')
ax3.set(xlabel='Dividing Wavelength [nm]', ylabel='Voc [V]')

#ax4.set_title('FF vs Dividing wavelenth')
#ax4.legend(loc='lower left')
ax4.set(xlabel='Dividing Wavelength [nm]', ylabel='Jsc [A/m^2]')

plt.show()