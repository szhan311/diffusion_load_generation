import matplotlib as mpl
import matplotlib.pyplot as plt

def hdr_plot_style():
    plt.style.use('dark_background')
    mpl.rcParams.update({'font.size': 18, 'lines.linewidth': 3, 'lines.markersize': 15})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Courier New'
    # mpl.rcParams['text.hinting'] = False
    # Set colors cycle
    colors = mpl.cycler('color', ['#3388BB', '#EE6666', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
    #plt.rc('figure', facecolor='#00000000', edgecolor='black')
    #plt.rc('axes', facecolor='#FFFFFF88', edgecolor='white', axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('legend', facecolor='#666666EE', edgecolor='white', fontsize=16)
    plt.rc('grid', color='white', linestyle='solid')
    plt.rc('text', color='white')
    plt.rc('xtick', direction='out', color='white')
    plt.rc('ytick', direction='out', color='white')
    plt.rc('patch', edgecolor='#E6E6E6')

def plot_diffusion(x_seq):
    interval = int(len(x_seq)/10)
    fig, axs = plt.subplots(2, 5, figsize=(28, 8))
    for i in range(10):
        cur_x = x_seq[i * interval]
        axs[int(i/5), i%5].plot(cur_x[0, :], linewidth=2);
        #axs[i].set_axis_off(); 
        axs[int(i/5), i%5].set_title('$q(\mathbf{x}_{'+str((i+1)*interval)+'})$')
    plt.tight_layout()

def plot_compare(ddpm, dataset, cond, test_id):
    cond_s = cond[test_id*28:(test_id+1)*28]
    x_seq = ddpm.sample_seq(batch_size=28, cond=cond_s)

    x_seq = x_seq.to("cpu")

    test_data = dataset[test_id*28:(test_id+1)*28].to('cpu')
    plt.figure(figsize=(15,3), dpi=300)
    plt.subplot(1,4,1)
    for i in range(len(test_data)):
        plt.plot(test_data[i])
    plt.title("actual data")
    plt.subplot(1,4,2)
    for i in range(len(x_seq[-1])):
        plt.plot(x_seq[-1][i])
    plt.title("generated data")
    plt.subplot(1,4,3)
    plt.plot(x_seq[-1].mean(dim=0), label = "mean of generated data")
    plt.plot(test_data.mean(dim=0), label = "mean of actual data")
    plt.legend(fontsize=10)
    plt.subplot(1,4,4)
    plt.plot(x_seq[-1].var(dim=0), label = "variance of generated data")
    plt.plot(test_data.var(dim=0), label = "variance of actual data")
    plt.legend(fontsize=10)
    plt.tight_layout()

# plot deterministic component
def plot_determ_comp(ddpm, cond, test_id):
    cond_s = cond[test_id*28:(test_id+1)*28]
    x_seq = ddpm.sample_seq(batch_size=28, cond=cond_s, stable=True)
    x_seq = x_seq.to("cpu")

    plt.figure(figsize=(8,4))
    for i in range(len(x_seq[-1])):
        plt.plot(x_seq[-1][i])
    plt.xlabel('time [30 min]')
    plt.ylabel("Load {}".format(test_id))
    plt.tight_layout()