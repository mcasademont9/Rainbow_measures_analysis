from tkinter.constants import N
from matplotlib import colors
import numpy as np
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plab
from scipy.interpolate import make_interp_spline


#This part is doing the same as before, but for the red cell. 
Cell_Files_Directory = filedialog.askdirectory(title='Select the directory of the RED CELL', initialdir=os.getcwd(), mustexist=True)
Cell_Files_Directory_rainbow = os.path.join(Cell_Files_Directory, 'TXT files')  # Ask for the directory where all the EQE and JV curve files are
Cell_Files_Directory_EQE = os.path.join(Cell_Files_Directory, 'EQE')
Cell_Available_folders_unsorted = os.listdir(Cell_Files_Directory_rainbow)
Cell_Available_folders = sorted(Cell_Available_folders_unsorted)
Cells_names = []
for folder in Cell_Available_folders:
    if not folder.endswith(".txt"):
        
        cell_pathlist = os.listdir(os.path.join(Cell_Files_Directory_rainbow,folder))
        for file in cell_pathlist:
            if file.endswith(".txt"):
                load_txt_filepath = os.path.join(Cell_Files_Directory_rainbow,folder,file)
                actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
                Cells_names.append(folder)
                try: 
                    Cell_megaarray
                except NameError: 
                    Cell_megaarray = actual_cell_data
                else: 
                    Cell_megaarray = np.dstack((Cell_megaarray, actual_cell_data))

Cell_Files_Directory_EQE_files = os.listdir(Cell_Files_Directory_EQE)
for file in Cell_Files_Directory_EQE_files:    
    for ii in range(len(Cells_names)):
        filename = str(Cells_names[ii]+'.txt')
        if file.endswith(filename):
            load_txt_filepath = os.path.join(Cell_Files_Directory_EQE,file)
            actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)            

            try: 
                Cell_EQE
            except NameError: 
                Cell_EQE = actual_cell_data
            else: 
                Cell_EQE = np.dstack((Cell_EQE, actual_cell_data))


#Plotting
gs = gridspec.GridSpec(3, 2)
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[2,:])
fig.suptitle('Rainbow measurements for '+os.path.split(Cell_Files_Directory)[1], fontsize=16)
colors_ = plab.cm.jet(np.linspace(0,1,len(Cell_megaarray[0,0,:])))
for i in range(len(Cell_megaarray[0,0,:])):
    ax1.plot(Cell_megaarray[:,0,i], Cell_megaarray[:,5,i], label = Cells_names[i], color=colors_[i])
    ax2.plot(Cell_megaarray[:,0,i], Cell_megaarray[:,4,i], label = Cells_names[i], color=colors_[i])
    ax3.plot(Cell_megaarray[:,0,i], Cell_megaarray[:,3,i], label = Cells_names[i], color=colors_[i])
    ax4.plot(Cell_megaarray[:,0,i], Cell_megaarray[:,2,i], label = Cells_names[i], color=colors_[i])
    x=Cell_EQE[:,0,i]
    y=Cell_EQE[:,1,i]
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(x.min(), 1100, 500)
    Y_ = X_Y_Spline(X_)
    ax5.plot(X_,Y_, label = Cells_names[i], color=colors_[i])
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

colors_ = plab.cm.jet(np.linspace(0,1,len(Cell_EQE[0,0,:])))
ax5.set(xlabel='Dividing Wavelength [nm]', ylabel='EQE [%]')
ax5.legend(loc='upper right')

plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)

Save_Directory = os.path.join(os.path.split(Cell_Files_Directory)[0],'Rainbow and EQE measurements plots')
if not os.path.exists(Save_Directory):
   os.mkdir(Save_Directory)

plt.savefig(os.path.join(os.path.split(Cell_Files_Directory)[0],'Rainbow and EQE measurements plots', os.path.split(Cell_Files_Directory)[1])+'.png')
print('Plot done!. It has been saved at '+str(os.path.join(os.path.split(Cell_Files_Directory)[0],'Rainbow and EQE measurements plots', os.path.split(Cell_Files_Directory)[1])+'.png'))