import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import argparse
import glob
import pickle
from collections import OrderedDict

from matplotlib import rc
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'weight':'bold'})
plt.rcParams.update({'axes.titlesize':24, 'axes.labelsize':24, 'legend.fontsize':18, 'xtick.labelsize':16, 'ytick.labelsize':14, 'axes.titleweight':'bold', 'axes.labelweight':'bold', 'figure.titleweight':'bold'})
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

parser = argparse.ArgumentParser(description='Plot Comparison of Syndrome Circuit vs Correlation Time and Noise Amplitude.')
parser.add_argument('directory', help='Directory containing the output pickle files.')
parser.add_argument('--savefig', help='If selected, the figure will be save in the output directory.', action='store_true')
parser.add_argument('--syndrome', help='Valid Entries are X or Z. This changes the titles of figures', default='')
args = parser.parse_args()

if args.syndrome is "X" or args.syndrome is "x":
    title = " X Syndrome"
elif args.syndrome is "Z" or args.syndrome is "z":
    title = " Z Syndrome"
else:
    title = ""


print(args.directory + "/*.p")
print(glob.glob(args.directory + "/*.p"))

pickle_files = glob.glob(args.directory + "/*.p")

saved_data = [{}]*len(pickle_files)
for i in range(len(pickle_files)):
    output = pickle.load(open(pickle_files[i], 'rb'))
    output['filename'] = pickle_files[i]
    saved_data[i] = output

# Find number of unique bandwidth values
power_values=set()
corr_values=set()
for data in saved_data:
    power_values.add(data['SPower'])
    corr_values.add(data['Bandwidth'])

corr_values = list(corr_values)
corr_values.sort(key=float)
power_values = list(power_values)
power_values.sort(key=float)
print(power_values)
print(corr_values)

power_plot_mezze_inf = OrderedDict()
power_plot_schwarma_inf = OrderedDict()
power_plot_schwarma_inf_stdev = OrderedDict()
corr_plot_mezze_inf = OrderedDict()
corr_plot_schwarma_inf = OrderedDict()
corr_plot_schwarma_inf_stdev = OrderedDict()
power_plot_x_axis = OrderedDict()
corr_plot_x_axis = OrderedDict()

for el in corr_values:
    power_plot_mezze_inf[el] = []
    power_plot_schwarma_inf[el] = []
    power_plot_schwarma_inf_stdev[el] = []
    power_plot_x_axis[el] = []
for el in power_values:
    corr_plot_mezze_inf[el] = []
    corr_plot_schwarma_inf[el] = []
    corr_plot_schwarma_inf_stdev[el] = []
    corr_plot_x_axis[el] = []

for data in saved_data:
    power_plot_mezze_inf[data["Bandwidth"]].append(data["FullSimInfidelity"])
    power_plot_schwarma_inf[data["Bandwidth"]].append(data["SchwARMAInfidelity"])
    power_plot_schwarma_inf_stdev[data["Bandwidth"]].append(np.std(data["AllSchwarmaInfidelities"]))
    power_plot_x_axis[data["Bandwidth"]].append(data["SPower"])
    corr_plot_mezze_inf[data["SPower"]].append(data["FullSimInfidelity"])
    corr_plot_schwarma_inf[data["SPower"]].append(data["SchwARMAInfidelity"])
    corr_plot_schwarma_inf_stdev[data["SPower"]].append(np.std(data["AllSchwarmaInfidelities"]))
    corr_plot_x_axis[data["SPower"]].append(data["Bandwidth"])

# Power Plot Ratio
plt.figure(1)
colors = cm.rainbow(np.linspace(0, 1, len(power_plot_mezze_inf)))
c=0
for key in power_plot_mezze_inf.keys():
    x_axis = power_plot_x_axis[key]
    y_data_mezze = power_plot_mezze_inf[key]
    y_data_schwarma = power_plot_schwarma_inf[key]
    y_data_ratio = np.array(power_plot_mezze_inf[key])/np.array(power_plot_schwarma_inf[key])
    plt.semilogx(x_axis, y_data_ratio, color=colors[c], linestyle='--', marker = 'o', label=str(key))
    c=c+1

plt.title('Fractional Simulation Error' + str(title))
plt.xlabel('Noise Power')
plt.ylabel('Mezze/SchWARMA Infidelity')
plt.legend()
#plt.show()
if args.savefig: plt.savefig(args.directory+"/ratio_vs_power.pdf")

# Bandwidth Plot Ratio
plt.figure(2)
colors = cm.rainbow(np.linspace(0, 1, len(corr_plot_mezze_inf)))
c=0
for key in corr_plot_mezze_inf.keys():
    x_axis = corr_plot_x_axis[key]
    y_data_mezze = corr_plot_mezze_inf[key]
    y_data_schwarma = corr_plot_schwarma_inf[key]
    y_data_ratio = np.array(corr_plot_mezze_inf[key]) / np.array(corr_plot_schwarma_inf[key])
    plt.semilogx(x_axis, y_data_ratio, color=colors[c], linestyle='--', marker = 'o', label=str(key), basex=2)
    c=c+1

plt.title('Fractional Simulation Error' + str(title))
plt.xlabel('Noise Correlation Time (Single Qubit Gate Lengths)')
plt.ylabel('Mezze/SchWARMA Infidelity')
plt.legend()
if args.savefig: plt.savefig(args.directory+"/ratio_vs_correlation.pdf")

# Power Plot Mezze
plt.figure(3)
colors = cm.rainbow(np.linspace(0, 1, len(power_plot_mezze_inf)))
c=0
for key in power_plot_mezze_inf.keys():
    x_axis = power_plot_x_axis[key]
    y_data_mezze = power_plot_mezze_inf[key]
    plt.loglog(x_axis, y_data_mezze, color=colors[c], linestyle='--', marker = 'o', label=str(key))
    c=c+1

plt.title('Trotter' + str(title))
plt.xlabel('Noise Power')
plt.ylabel('Trotter Infidelity')
plt.legend()
if args.savefig: plt.savefig(args.directory+"/trotter_vs_power.pdf")

# Mezze Bandwidth
plt.figure(4)
colors = cm.rainbow(np.linspace(0, 1, len(corr_plot_mezze_inf)))
c=0
for key in corr_plot_mezze_inf.keys():
    x_axis = corr_plot_x_axis[key]
    y_data_mezze = corr_plot_mezze_inf[key]
    plt.loglog(x_axis, y_data_mezze, color=colors[c], linestyle='--', marker = 'o', label=str(key), basex=2)
    c=c+1

plt.title('Trotter'  + str(title))
plt.xlabel('Noise Correlation Time (Single Qubit Gate Lengths)')
plt.ylabel('Trotter Infidelity')
plt.legend()
if args.savefig: plt.savefig(args.directory+"/trotter_vs_correlation.pdf")

# SchWARMA Power Plot
plt.figure(5)
colors = cm.rainbow(np.linspace(0, 1, len(power_plot_mezze_inf)))
c=0
for key in power_plot_mezze_inf.keys():
    x_axis = power_plot_x_axis[key]
    y_data_schwarma = power_plot_schwarma_inf[key]
    y_err_schwarma = power_plot_schwarma_inf_stdev[key]
    plt.loglog(x_axis, y_data_schwarma, color=colors[c], marker='p', label=str(key))
    c=c+1

plt.title('SchWARMA'  + str(title))
plt.xlabel('Noise Power')
plt.ylabel('SchWARMA Infidelity')
plt.legend()
if args.savefig: plt.savefig(args.directory+"/schwarma_vs_power.pdf")

# SchWARMA Bandwidth
plt.figure(6)
colors = cm.rainbow(np.linspace(0, 1, len(corr_plot_mezze_inf)))
c=0
for key in corr_plot_mezze_inf.keys():
    x_axis = corr_plot_x_axis[key]
    y_data_schwarma = corr_plot_schwarma_inf[key]
    y_err_schwarma = corr_plot_schwarma_inf_stdev[key]
    plt.loglog(x_axis, y_data_schwarma, color=colors[c], marker='p', label=str(key), basex=2)
    c=c+1

plt.title('SchWARMA' + str(title))
plt.xlabel('Noise Correlation Time (Single Qubit Gate Lengths)')
plt.ylabel('SchWARMA Infidelity')
plt.legend()
if args.savefig: plt.savefig(args.directory+"/schwarma_vs_correlation.pdf")

# SchWARMA/Mezze Power Plot
plt.figure(7)
colors = cm.rainbow(np.linspace(0, 1, len(power_plot_mezze_inf)))
c=0
for key in power_plot_mezze_inf.keys():
    x_axis = power_plot_x_axis[key]
    y_data_mezze = power_plot_mezze_inf[key]
    y_data_schwarma = power_plot_schwarma_inf[key]
    y_err_schwarma = power_plot_schwarma_inf_stdev[key]
    plt.loglog(x_axis, y_data_mezze, color=colors[c], linestyle='--', marker = 'o', label=str(key)+str(" Trotter"))
    plt.loglog(x_axis, y_data_schwarma, color=colors[c], marker='p', label=str(key)+str(" SchWARMA"))
    c=c+1

plt.title('Trotter/SchWARMA Comparison'  + str(title))
plt.xlabel('Noise Power')
plt.ylabel('Infidelity')
plt.legend(loc='best', fontsize=6)
if args.savefig: plt.savefig(args.directory+"/comparison_vs_power.pdf")

# Schwarma/Mezze Bandwidth
plt.figure(8)
colors = cm.rainbow(np.linspace(0, 1, len(corr_plot_mezze_inf)))
c=0
for key in corr_plot_mezze_inf.keys():
    x_axis = corr_plot_x_axis[key]
    y_data_mezze = corr_plot_mezze_inf[key]
    y_data_schwarma = corr_plot_schwarma_inf[key]
    y_err_schwarma = corr_plot_schwarma_inf_stdev[key]
    plt.loglog(x_axis, y_data_mezze, color=colors[c], linestyle='--', marker = 'o', label=str(key)+str(" Trotter"), basex=2)
    plt.loglog(x_axis, y_data_schwarma, color=colors[c], marker='p', label=str(key)+str(" SchWARMA"), basex=2)
    c=c+1

plt.title('Trotter/SchWARMA Comparison' + str(title))
plt.xlabel('Noise Correlation Time (Single Qubit Gate Lengths)')
plt.ylabel('Infidelity')
if args.savefig: plt.savefig(args.directory+"/comparison_vs_correlation.pdf")
#plt.show()

# Schwarma/Mezze Bandwidth Errorbar
plt.figure(9)
colors = cm.rainbow(np.linspace(0, 1, len(corr_plot_mezze_inf)))
c=0
for key in corr_plot_mezze_inf.keys():
    x_axis = corr_plot_x_axis[key]
    y_data_mezze = corr_plot_mezze_inf[key]
    y_data_schwarma = corr_plot_schwarma_inf[key]
    y_err_schwarma = 0.434 * np.array(corr_plot_schwarma_inf_stdev[key]) / np.array(corr_plot_schwarma_inf[key])
    plt.xscale("log", nonposx='clip', basex=2)
    plt.plot(x_axis, np.log10(y_data_mezze), color=colors[c], linestyle='--', marker = 'o', label=str(key)+str(" Trotter"))
    plt.errorbar(x_axis, np.log10(y_data_schwarma), yerr = y_err_schwarma, color=colors[c], marker='p', label=str(key)+str(" SchWARMA"), capsize=3)
    c=c+1

ylim = plt.ylim()
plt.ylim([np.min(ylim), 0])
plt.title(str(title))
plt.xlabel('Correlation Time (Gate Lengths)')
plt.ylabel('Infidelity')
locs, labels = plt.yticks()
plt.yticks(locs, ['$10^{'+str(int(loc))+'}$' for loc in locs])
plt.tight_layout()
if args.savefig: plt.savefig(args.directory+"/comparison_vs_correlation_errorbar.pdf")
plt.show()
