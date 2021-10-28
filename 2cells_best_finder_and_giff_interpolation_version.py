# This script is taking two sets of rainbow measures (red sweep and blue sweep) and compares all the cells in order to find the best wavelength cut for each pair of cells.
from tkinter.constants import N, TRUE
from matplotlib import colors

# import Small_Functions as sf
import numpy as np
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plab
import imageio
from scipy import interpolate
from matplotlib import colors


################################################
########### ANALYSIS TYPE SELECTOR #############
################################################

BEST_N_CELLS_TF = (
    False  # If True, it returns the PCE graphs of the N bests binary rainbow cells
)
N_cells = 50  # The number of best cells to return for the BEST_N_CELLS option


GIF_TF = False  # If true, it returns a folder with all the GIFs of each Blue/Red cell swaping with the others Red/Blue cells.
RED_CELL_CONSTANT = False  # If True, it makes the giff with the Red cell constant
BLUE_CELL_CONSTANT = True  # If True, it makes the giff with the Blue cell constant
# NOTE: Both Blue and Red cell constant options can be true. In that case, it will create both folders.
# Consider also that this GIFF option is highly time consuming, so just use it in case you really need it (for example, it is cool to see the thickness dependence of IoBC)

FULL_RAINBOW_EQE_PLOTS_TF = True  # If True it returns a graphic for Red and Blue cells wherein the rainbow measurements of all the cells
AVAILABLE_EQE_DATA = False  # The False option creates the FULL_RAINBOW_EQE_PLOTS_TF plot withouth the EQE. Use it in case you don't have the EQE data or if you don't want to show it.

################################################
########### DATA STRUCURE AND OUTPUT ###########
################################################

# Your data should be structured as follows:
#
# 1 - One main folder with the default structure of the rainbow measure: DeviceName > 'TXT Files' > CellNumber > RainbowMeasure.txt
# It is important that only one RainbowMeasure.txt file exists in each CellNumber folder (the one corresponding to the blue or red sweep, maybe specified in the DeviceName folder)
# It is recomended that both blue and red cells main folder (DeviceName folder) are placed in the same path, in this way the output is easier to find.
#
# 2- If you have EQE data for each cell, the default folder EQE setup should be placed inside the DeviceName folder with the name EQE.
# It is also very important that the name of the EQE file of each CellNumber contains the same string as CellNumber (in principle it is done by default, but just keep in mind).
# Therefore, the EQE of the a concret CellNumber should be place at: DeviceName > 'EQE' > EQE_CellNumber_EQE.txt
# The EQE folder can also contain other foleders and other TXT files that does not have any CellNumber inside its name.
#
# 3- The outputs of BEST_N_CELLS_TF and GIF_TF will be stored in a folder inside the path of DeviceName of both Red and Blue (recomended to be the same) called 'Rainbow analysis (2 cells)'
# separeted for each Red/Blue device combination.
#
# 4- The outputs of FULL_RAINBOW_EQE_PLOTS_TF will be stored in a folder inside the path of DeviceName of both Red and Blue (recomended to be the same) called 'Rainbow and EQE measurements plots'

################################################
################################################
################################################


# First of all we load all the dataof red and blue cells. For that, we ask to the user where are the measure files of the Blue cell and the Red cell,
# read each rainbow measure for each measured cell and save it on an array (BlueCell_megaarray, RecCell_megaarray) and its corresponding _names in a list.


BlueCell_Files_Directory = filedialog.askdirectory(
    title="Select the directory of the BLUE CELL",
    initialdir=os.getcwd(),
    mustexist=True,
)  # Ask for the directory where BlueCell is allocated (DeviceName)
BlueCell_Files_Directory_rainbow = os.path.join(
    BlueCell_Files_Directory, "TXT files"
)  # Define the directory where all the rainbow sweeps are saved (CellNumber)
BlueCell_Available_folders = os.listdir(BlueCell_Files_Directory_rainbow)
BlueCell_device_name = os.path.split(BlueCell_Files_Directory)[1]
BlueCells_names = []  # initialize the list of names
for folder in BlueCell_Available_folders:
    if not folder.endswith(
        ".txt"
    ):  # find for all the folders inside the specified path which corresponds to the measured cells without reading the calibration files (which are .txt files)
        cell_pathlist = os.listdir(
            os.path.join(BlueCell_Files_Directory_rainbow, folder)
        )  # enter to each foler corresponding to each measured cell
        for file in cell_pathlist:
            if file.endswith(
                ".txt"
            ):  # And store the data in the corresponding megaarray
                load_txt_filepath = os.path.join(
                    BlueCell_Files_Directory_rainbow, folder, file
                )
                actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
                BlueCells_names.append(folder)
                try:
                    BlueCell_megaarray
                except NameError:
                    BlueCell_megaarray = actual_cell_data
                else:
                    BlueCell_megaarray = np.dstack(
                        (BlueCell_megaarray, actual_cell_data)
                    )

if AVAILABLE_EQE_DATA:
    BlueCell_Files_Directory_EQE = os.path.join(
        BlueCell_Files_Directory, "EQE"
    )  # Define the directory where all the EQE data is saved
    for file in BlueCell_Files_Directory_EQE:
        for ii in range(len(BlueCells_names)):
            if file.endswith(BlueCells_names[ii] + ".txt"):
                load_txt_filepath = os.path.join(BlueCell_Files_Directory_EQE, file)
                actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)

                try:
                    BlueCell_EQE
                except NameError:
                    BlueCell_EQE = actual_cell_data
                else:
                    BlueCell_EQE = np.dstack((BlueCell_EQE, actual_cell_data))

# This part is doing the same as before, but for the red cell.
RedCell_Files_Directory = filedialog.askdirectory(
    title="Select the directory of the RED CELL", initialdir=os.getcwd(), mustexist=True
)  # Ask for the directory where RedCell is allocated (DeviceName)
RedCell_Files_Directory_rainbow = os.path.join(
    RedCell_Files_Directory, "TXT files"
)  # Define the directory where all the rainbow sweeps are saved (CellNumber)
RedCell_Available_folders = os.listdir(RedCell_Files_Directory_rainbow)
RedCell_device_name = os.path.split(RedCell_Files_Directory)[1]
RedCells_names = []
for folder in RedCell_Available_folders:
    if not folder.endswith(".txt"):

        cell_pathlist = os.listdir(
            os.path.join(RedCell_Files_Directory_rainbow, folder)
        )
        for file in cell_pathlist:
            if file.endswith(".txt"):
                load_txt_filepath = os.path.join(
                    RedCell_Files_Directory_rainbow, folder, file
                )
                actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
                RedCells_names.append(folder)
                try:
                    RedCell_megaarray
                except NameError:
                    RedCell_megaarray = actual_cell_data
                else:
                    RedCell_megaarray = np.dstack((RedCell_megaarray, actual_cell_data))

if AVAILABLE_EQE_DATA:
    RedCell_Files_Directory_EQE = os.path.join(RedCell_Files_Directory, "EQE")
    for file in RedCell_Files_Directory_EQE:
        for ii in range(len(RedCells_names)):
            if file.endswith(RedCells_names[ii] + ".txt"):
                load_txt_filepath = os.path.join(RedCell_Files_Directory_EQE, file)
                actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)
                try:
                    RedCell_EQE
                except NameError:
                    RedCell_EQE = actual_cell_data
                else:
                    RedCell_EQE = np.dstack((RedCell_EQE, actual_cell_data))

print("Blue cell and Red cell data loaded successfully")

# Here we make the arrays corresponding to all the possible sums between cells.
# Fist of all we will re-create the megaarrays so that we have a better cutwl resolution making and interpolation between them.
BlueCell_megaarray_raw = BlueCell_megaarray  # We first save the imported data stored in the megaarays as a new variable called megaarray_raw
RedCell_megaarray_raw = RedCell_megaarray
BlueCell_megaarray = None  # And kill the old megaarrays. This is because I made this part of the code afterwards and I dont want to change evey name of all the megaarrays.
RedCell_megaarray = None
megaarray_minwl = max(
    (np.min(RedCell_megaarray_raw[:, 0, :]), np.min(BlueCell_megaarray_raw[:, 0, :]))
)
megaarray_maxwl = min(
    (np.max(RedCell_megaarray_raw[:, 0, :]), np.max(BlueCell_megaarray_raw[:, 0, :]))
)
print(megaarray_maxwl, megaarray_minwl)
megaarray_reswl = 1
megaarray_wls = np.arange(megaarray_minwl, megaarray_maxwl, megaarray_reswl)
BlueCell_megaarray = np.zeros(
    (
        len(megaarray_wls),
        len(BlueCell_megaarray_raw[0, :, 0]),
        len(BlueCell_megaarray_raw[0, 0, :]),
    )
)
RedCell_megaarray = np.zeros(
    (
        len(megaarray_wls),
        len(RedCell_megaarray_raw[0, :, 0]),
        len(RedCell_megaarray_raw[0, 0, :]),
    )
)


for j in range(
    len(BlueCell_megaarray[0, 0, :])
):  # We loop for all the cells (the deep dimension)
    for i in range(
        len(BlueCell_megaarray_raw[0, :])
    ):  # Now we loop for all the columns of the megaarrays_raw that contains information (ie, not the first that is cut wl)
        if i == 0:
            BlueCell_megaarray[:, 0, j] = megaarray_wls

        blue_interp = interpolate.interp1d(
            BlueCell_megaarray_raw[:, 0, j],
            BlueCell_megaarray_raw[:, i, j],
            kind="linear",
        )  # We make a linear interpolation of each column
        BlueCell_megaarray[:, i, j] = blue_interp(
            BlueCell_megaarray[:, 0, j]
        )  # And save it to the corresponding column with the wl of the sero column (the wl resolution that we want)

for j in range(
    len(RedCell_megaarray[0, 0, :])
):  # We loop for all the cells (the deep dimension)
    for i in range(
        len(BlueCell_megaarray_raw[0, :])
    ):  # Now we loop for all the columns of the megaarrays_raw that contains information (ie, not the first that is cut wl)
        if i == 0:
            RedCell_megaarray[:, 0, j] = megaarray_wls

        red_interp = interpolate.interp1d(
            RedCell_megaarray_raw[:, 0, j],
            RedCell_megaarray_raw[:, i, j],
            kind="linear",
        )  # We make a linear interpolation of each column
        RedCell_megaarray[:, i, j] = red_interp(
            RedCell_megaarray[:, 0, j]
        )  # And save it to the corresponding column with the wl of the sero column (the wl resolution that we want)


Sums_names = []  # first start the list of cell names for each sum
for i in range(len(BlueCell_megaarray[0, 0, :])):  # Loop for the blue cell
    for j in range(len(RedCell_megaarray[0, 0, :])):  # Loop for the red cell
        Summ_actual_array = np.zeros(
            RedCell_megaarray[:, :, j].shape
        )  # first initialize the numpy array of the actual summ

        for r_wl_ind in range(
            len(RedCell_megaarray[:, 0, j])
        ):  # This small part is added just in case that red and blue sweep does not have the same cutting wavelenght order (it seareches for the corresponding cutting wavelenght of the red array to the blue array and makes the summ)
            b_wl_ind = np.where(
                BlueCell_megaarray[:, 0, i] == RedCell_megaarray[r_wl_ind, 0, j]
            )[
                0
            ]  # Find the index of the blue cell array that has the same dividing wavelength as the red cell array
            # print(i,j)
            Summ_actual_array[r_wl_ind, :] = (
                BlueCell_megaarray[b_wl_ind, :, i] + RedCell_megaarray[r_wl_ind, :, j]
            )  # For that wavelength, sum of all the parameters
            Summ_actual_array[r_wl_ind, 0:1] = (
                Summ_actual_array[r_wl_ind, 0:1] / 2
            )  # Except the dividing wavelength (which should be the same for all the data) and sun power (which will be similar, so we make the average)

        Summ_actual_name = (
            BlueCells_names[i] + "Blue cell / " + RedCells_names[j] + "Red cell"
        )  # Saving the name of the corresponding cell pairs
        Summ_actual_index = np.array(
            [i, j]
        )  # and the index of the BlueCell and RedCell megaarrays so afterwards se can plot them

        if i == 0 and j == 0:  # Store the data in new arrays for the first iteration
            Summ_megaarray = Summ_actual_array
            Sums_names.append(Summ_actual_name)
            Summs_indexes = Summ_actual_index

        else:
            Summ_megaarray = np.dstack(
                (Summ_megaarray, Summ_actual_array)
            )  # and stacking them for the nexts iterations
            Sums_names.append(Summ_actual_name)
            Summs_indexes = np.vstack((Summs_indexes, Summ_actual_index))


# Now we will calculate the IOBC (improvement over best cell) for each pair of cells and the maximum PCE that each combination achieves
IoBC = (
    []
)  # First start the list that will have all the IOBC values for each combination
maxPCE = []  # And the PCE_max values
for n in range(
    0, len(Summ_megaarray[0, 0, :])
):  # Loop arround all the possible combinations
    [i, j] = Summs_indexes[n, :]
    best_cell_PCE = np.amax(
        (RedCell_megaarray[:, 5, j], BlueCell_megaarray[:, 5, i])
    )  # And find the maxim efficiency of blue and red cell
    IoBC_actual = 100 * (
        (np.amax(Summ_megaarray[:, 5, n]) / best_cell_PCE) - 1
    )  # Afterwards, calculate the IOBC as the division of the maximum of the summ divided by the best cell PCE and rest 1
    IoBC.append(IoBC_actual)  # Finally put the actual IOBC in the IOBC list
    maxPCE.append(best_cell_PCE)


# We add a IoBC treshold, since tanking the best and the worts cell will give something with a really high maxPCE but a IoBC near to 0
# for t in range(len(IoBC)):
#     if IoBC[t] <= 8:
#         IoBC[t] = 0
#         maxPCE[t] = 0

if BEST_N_CELLS_TF:

    # Now find the N combinations with higher IOBC, and appart the N with higher total PCE
    # N_cells = 20
    IoBC_max = np.argsort(
        np.array(IoBC)
    )  # [-N:]            #Find the corresponding indexes in the sum matrixes corresponding to N maximum IoBC
    # maxPCE_max = np.argsort(np.array(maxPCE))#[-N:]        #Find the corresponding indexes in the sum matrixes corresponding to N maximum maxPCE
    # IoBC_max_and_maxPCE_max = np.intersect1d(IoBC_max, maxPCE_max) #Find the indexes that are to both cases
    # IoBC_max_and_maxPCE_max = np.array([IoBC,maxPCE], dtype=[('IoBC_max', 'f4'),('maxPCE','f4')])
    # IoBC_max_and_maxPCE_max = np.argsort(IoBC_max_and_maxPCE_max, order=('maxPCE', 'IoBC_max'))

    gs = gridspec.GridSpec(1, 1)
    for k in range(
        N_cells
    ):  # Now we plot all the results for the N best cells combinations
        n = IoBC_max[-k - 1]
        [i, j] = Summs_indexes[n, :]
        xred = RedCell_megaarray[:, 0, j]
        yred = RedCell_megaarray[:, 5, j]
        xblue = BlueCell_megaarray[:, 0, i]
        yblue = BlueCell_megaarray[:, 5, i]
        xtot = Summ_megaarray[:, 0, n]
        ytot = Summ_megaarray[:, 5, n]
        summ_max_index = np.amax(ytot)
        fig = plt.figure(figsize=(8, 5))
        fig.suptitle(
            "Rainbow analysis "
            + BlueCell_device_name
            + " (Blue cell) "
            + BlueCells_names[i]
            + "/"
            + RedCell_device_name
            + "(Red cell) "
            + RedCells_names[j],
            fontsize=12,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(xred, yred, color="tab:red", label=RedCells_names[j] + "Red Cell")
        ax1.plot(xblue, yblue, color="tab:blue", label=BlueCells_names[i] + "Blue Cell")
        ax1.plot(xtot, ytot, color="tab:green", label="Red cell + Blue Cell")
        ax1.set_title(
            "Rainbow PCE(%)="
            + str(format(maxPCE[n] * 0.01 * (100 + IoBC[n]), ".2f"))
            + "  Red/Blue-PCE(%)="
            + str(format(np.amax(RedCell_megaarray[:, 5, j]), ".2f"))
            + "/"
            + str(format(np.amax(BlueCell_megaarray[:, 5, i]), ".2f"))
            + "   IoBC(%)="
            + str(format(IoBC[n], ".2f"))
        )
        ax1.legend(loc="lower center")
        ax1.set(xlabel="Dividing Wavelength /nm", ylabel="PCE /%")
        filename = "/Best combination_" + str(k + 1) + ".png"
        Save_Directory = os.path.join(
            os.path.split(BlueCell_Files_Directory)[0],
            "Rainbow analysis (2 cells)",
            BlueCell_device_name + " + " + RedCell_device_name,
            str(N_cells) + " best cells combinations",
        )
        if not os.path.exists(Save_Directory):
            os.makedirs(Save_Directory)
        plt.savefig(str(Save_Directory + filename))


if GIF_TF:

    # Creating the gifs
    # Creating the frames
    filenames = []
    gs = gridspec.GridSpec(1, 1)
    ymax = max(maxPCE)
    for j in range(len(RedCell_megaarray[0, 0, :])):
        for i in range(
            len(BlueCell_megaarray[0, 0, :])
        ):  # Now we plot all the results for the N best cells combinations
            nn = np.intersect1d(
                np.argwhere(Summs_indexes[:, 0] == i),
                np.argwhere(Summs_indexes[:, 1] == j),
            )
            xred = RedCell_megaarray[:, 0, j]
            yred = RedCell_megaarray[:, 5, j]
            xblue = BlueCell_megaarray[:, 0, i]
            yblue = BlueCell_megaarray[:, 5, i]
            xtot = Summ_megaarray[:, 0, nn[0]]
            ytot = Summ_megaarray[:, 5, nn[0]]
            # IoBC = 100*(np.amax(ytot)/(np.amax((yred, yblue)))-1)

            fig = plt.figure(figsize=(8, 5))
            fig.suptitle(
                "Rainbow analysis "
                + BlueCell_device_name
                + " (Blue cell) "
                + BlueCells_names[i]
                + "/"
                + RedCell_device_name
                + "(Red cell) "
                + RedCells_names[j],
                fontsize=12,
            )
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(xred, yred, color="tab:red", label=RedCells_names[j] + " Red Cell")
            ax1.plot(
                xblue, yblue, color="tab:blue", label=BlueCells_names[i] + " Blue Cell"
            )
            ax1.plot(xtot, ytot, color="tab:green", label="Red cell + Blue Cell")
            ax1.set_title(
                "Efficiency vs Dividing Wavelength. IOBC(%)="
                + str(format(IoBC[nn[0]], ".4f"))
            )
            ax1.legend(loc="upper right")
            ax1.set_ylim(0, ymax * 1.05)
            ax1.set(xlabel="Dividing Wavelength /nm", ylabel="Efficiency /%")
            Save_Directory = os.path.join(
                os.path.split(BlueCell_Files_Directory)[0],
                "Rainbow analysis (2 cells)",
                BlueCell_device_name + " + " + RedCell_device_name,
                "GIFs",
            )
            if not os.path.exists(Save_Directory):
                os.makedirs(Save_Directory)
            filename = Save_Directory + "/" + str(i) + ".png"
            plt.savefig(str(filename))
            filenames.append(filename)

        # Build GIF
        frames = []
        for filename in filenames:
            frames.append(imageio.imread(filename))

        # Save GIF
        exportname = (
            Save_Directory
            + "/"
            + RedCell_device_name
            + "_"
            + RedCells_names[j]
            + "_"
            + BlueCell_device_name
            + ".gif"
        )
        kargs = {"duration": 1.5}  # Time in secons that each frame lasts in the screen
        imageio.mimsave(exportname, frames, "GIF", **kargs)

        # Remove frame files
        for filename in set(filenames):
            os.remove(filename)

if FULL_RAINBOW_EQE_PLOTS_TF:

    # Now create the plots of the rainbow measures and the EQEs
    # This part was an old part which directly read again all the data, but I want to modify it so that it does not reload the data because we have this data already loaded.
    # RedBlue_paths = (RedCell_Files_Directory, BlueCell_Files_Directory)
    # for pth in range(
    #     len(RedBlue_paths)
    # ):  # Here we load again the data for each cell and its EQE

    #     Cell_Files_Directory = RedBlue_paths[pth]
    #     Cell_Files_Directory_rainbow = os.path.join(
    #         Cell_Files_Directory, "TXT files"
    #     )  # Ask for the directory where all the EQE and JV curve files are
    #     Cell_Files_Directory_EQE = os.path.join(Cell_Files_Directory, "EQE")
    #     Cell_Available_folders = os.listdir(Cell_Files_Directory_rainbow)
    #     Cells_names = []
    #     for folder in Cell_Available_folders:
    #         if not folder.endswith(".txt"):
    #             Cells_names.append(folder)
    #             cell_pathlist = os.listdir(
    #                 os.path.join(Cell_Files_Directory_rainbow, folder)
    #             )
    #             for file in cell_pathlist:
    #                 if file.endswith(".txt"):
    #                     load_txt_filepath = os.path.join(
    #                         Cell_Files_Directory_rainbow, folder, file
    #                     )
    #                     actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)

    #                     try:
    #                         Cell_megaarray
    #                     except NameError:
    #                         Cell_megaarray = actual_cell_data
    #                     else:
    #                         Cell_megaarray = np.dstack(
    #                             (Cell_megaarray, actual_cell_data)
    #                         )

    #     Cell_Files_Directory_EQE_files = os.listdir(Cell_Files_Directory_EQE)
    #     for file in Cell_Files_Directory_EQE_files:
    #         for ii in range(len(Cells_names)):
    #             filename = str(Cells_names[ii] + ".txt")
    #             if file.endswith(filename):
    #                 load_txt_filepath = os.path.join(Cell_Files_Directory_EQE, file)
    #                 actual_cell_data = np.loadtxt(load_txt_filepath, skiprows=1)

    #                 try:
    #                     Cell_EQE
    #                 except NameError:
    #                     Cell_EQE = actual_cell_data
    #                 else:
    #                     Cell_EQE = np.dstack((Cell_EQE, actual_cell_data))
    ############################################################################
    # Here starts the new code!

    RedBlue_paths = (RedCell_Files_Directory, BlueCell_Files_Directory)
    for i in range(len(RedBlue_paths)):
        Cell_Files_Directory = RedBlue_paths[i]
        if i == 0:
            Cell_megaarray = RedCell_megaarray_raw
            Cells_names = RedCells_names
            if AVAILABLE_EQE_DATA:
                Cell_EQE = RedCell_EQE

        if i == 1:
            Cell_megaarray = BlueCell_megaarray_raw
            Cells_names = BlueCells_names
            if AVAILABLE_EQE_DATA:
                Cell_EQE = BlueCell_EQE
        # Plotting
        if AVAILABLE_EQE_DATA:
            gs = gridspec.GridSpec(3, 2)  # We define a grid with 3 rows and 2 columns
            fig = plt.figure(
                figsize=(12, 12)
            )  # After some testing, this figsize is the best (12,12)
            ax1 = fig.add_subplot(
                gs[0, 0]
            )  # we plot the different solar cell parameters in the 2x2 top matrix
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            ax5 = fig.add_subplot(
                gs[2, :]
            )  # And the EQE will be plotted as a single plot at the bottom in both two columns
            fig.suptitle(
                "Rainbow measurements for " + os.path.split(Cell_Files_Directory)[1],
                fontsize=16,
            )  # We define the title
            colors_ = plab.cm.jet(
                np.linspace(0, 1, len(Cell_megaarray[0, 0, :]))
            )  # Add the gradient color so that the gradient thickess has some sense on the eyes
            for nn in range(
                len(Cell_megaarray[0, 0, :])
            ):  # And loop for all the cells of the megaarray to plot the different OSC parameters and EQE
                ax1.plot(
                    Cell_megaarray[:, 0, nn],
                    Cell_megaarray[:, 5, nn],
                    label=Cells_names[nn],
                    color=colors_[n],
                )
                ax2.plot(
                    Cell_megaarray[:, 0, nn],
                    Cell_megaarray[:, 4, nn],
                    label=Cells_names[nn],
                    color=colors_[nn],
                )
                ax3.plot(
                    Cell_megaarray[:, 0, nn],
                    Cell_megaarray[:, 3, nn],
                    label=Cells_names[nn],
                    color=colors_[nn],
                )
                ax4.plot(
                    Cell_megaarray[:, 0, nn],
                    Cell_megaarray[:, 2, nn],
                    label=Cells_names[nn],
                    color=colors_[nn],
                )
                x = Cell_EQE[:, 0, nn]
                y = Cell_EQE[:, 1, nn]
                X_Y_Spline = interpolate.make_interp_spline(
                    x, y
                )  # Here I make an interpolation to the EQE
                X_ = np.linspace(
                    x.min(), x.max(), len(Cell_EQE)
                )  # So that if needed, one can choose the min and max wavelengths where EQE is ploted, as well as the resolution. By default is set to the same as measured
                Y_ = X_Y_Spline(X_)
                ax5.plot(X_, Y_, label=Cells_names[nn], color=colors_[nn])

            # ax1.set_title('PCE vs Dividing wavelenth')
            if i == 0:
                ax1.legend(loc="upper right")

            if i == 1:
                ax1.legend(loc="upper left")
            ax1.set(xlabel="Dividing Wavelength [nm]", ylabel="Efficiency [%]")

            # ax2.set_title('FF vs Dividing wavelenth')
            # ax2.legend(loc='lower left')
            ax2.set(xlabel="Dividing Wavelength [nm]", ylabel="FF [%]")

            # ax3.set_title('FF vs Dividing wavelenth')
            # ax3.legend(loc='lower left')
            ax3.set(xlabel="Dividing Wavelength [nm]", ylabel="Voc [V]")

            # ax4.set_title('FF vs Dividing wavelenth')
            # ax4.legend(loc='lower left')
            ax4.set(xlabel="Dividing Wavelength [nm]", ylabel="Jsc [A/m^2]")

            colors_ = plab.cm.jet(np.linspace(0, 1, len(Cell_EQE[0, 0, :])))
            ax5.set(xlabel="Wavelength [nm]", ylabel="EQE [%]")
            ax5.legend(loc="upper right")

            plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)

            Save_Directory = os.path.join(
                os.path.split(Cell_Files_Directory)[0],
                "Rainbow and EQE measurements plots",
            )
            if not os.path.exists(Save_Directory):
                os.mkdir(Save_Directory)

            plt.savefig(
                os.path.join(
                    os.path.split(Cell_Files_Directory)[0],
                    "Rainbow and EQE measurements plots",
                    os.path.split(Cell_Files_Directory)[1],
                )
                + ".png"
            )

        if not AVAILABLE_EQE_DATA:
            gs = gridspec.GridSpec(2, 2)
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            fig.suptitle(
                "Rainbow measurements for " + os.path.split(Cell_Files_Directory)[1],
                fontsize=16,
            )
            colors_ = plab.cm.jet(np.linspace(0, 1, len(Cell_megaarray[0, 0, :])))
            for nn in range(len(Cell_megaarray[0, 0, :])):
                ax1.plot(
                    Cell_megaarray[:, 0, nn],
                    Cell_megaarray[:, 5, nn],
                    label=Cells_names[nn],
                    color=colors_[nn],
                )
                ax2.plot(
                    Cell_megaarray[:, 0, nn],
                    Cell_megaarray[:, 4, nn],
                    label=Cells_names[nn],
                    color=colors_[nn],
                )
                ax3.plot(
                    Cell_megaarray[:, 0, nn],
                    Cell_megaarray[:, 3, nn],
                    label=Cells_names[nn],
                    color=colors_[nn],
                )
                ax4.plot(
                    Cell_megaarray[:, 0, nn],
                    Cell_megaarray[:, 2, nn],
                    label=Cells_names[nn],
                    color=colors_[nn],
                )

            # ax1.set_title('PCE vs Dividing wavelenth')
            if i == 0:
                ax1.legend(loc="upper right")

            if i == 1:
                ax1.legend(loc="upper left")

            ax1.set(xlabel="Dividing Wavelength [nm]", ylabel="Efficiency [%]")

            # ax2.set_title('FF vs Dividing wavelenth')
            # ax2.legend(loc='lower left')
            ax2.set(xlabel="Dividing Wavelength [nm]", ylabel="FF [%]")

            # ax3.set_title('FF vs Dividing wavelenth')
            # ax3.legend(loc='lower left')
            ax3.set(xlabel="Dividing Wavelength [nm]", ylabel="Voc [V]")

            # ax4.set_title('FF vs Dividing wavelenth')
            # ax4.legend(loc='lower left')
            ax4.set(xlabel="Dividing Wavelength [nm]", ylabel="Jsc [A/m^2]")

            plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)

            Save_Directory = os.path.join(
                os.path.split(Cell_Files_Directory)[0],
                "Rainbow and EQE measurements plots",
            )
            if not os.path.exists(Save_Directory):
                os.mkdir(Save_Directory)

            plt.savefig(
                os.path.join(
                    os.path.split(Cell_Files_Directory)[0],
                    "Rainbow and EQE measurements plots",
                    os.path.split(Cell_Files_Directory)[1],
                )
                + ".png"
            )
