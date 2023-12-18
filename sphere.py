import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.interpolate import interp1d
from datetime import datetime
from tqdm import trange

n = 64
T = 15
dt = 0.1
num_steps = int(T/dt) + 1
d = 3
beta = 1
denominator = True
half_sphere = False

V = np.eye(d)
A = np.eye(d)

x0 = np.random.randn(n, d)
x0 /= np.linalg.norm(x0, axis=1)[:, np.newaxis]

z0 = np.random.randn(n, d)
z0 /= np.linalg.norm(z0, axis=1)[:, np.newaxis]

if half_sphere:
    # Ensuring z-values are non-negative to lie on the upper half-sphere
    x0[x0[:, 2] < 0] *= -1

z = np.zeros(shape=(n, num_steps, d)) 
z[:, 0, :] = x0
integration_time = np.linspace(0, T, num_steps)

for l, t in enumerate(integration_time[:-1]):
    Az = np.matmul(A, z[:, l, :].T)
    exp_beta_dot = np.exp(beta * np.matmul(Az.T, Az))
    if denominator:
        attention = exp_beta_dot / exp_beta_dot.sum(axis=1)[:, np.newaxis]
    else:
        attention = exp_beta_dot / n
    
    dlst = np.matmul(attention, np.matmul(V, z[:, l, :].T).T)
    dynamics = dlst
    z[:, l+1, :] = z[:, l, :] + dt * dynamics
    z[:, l+1, :] = z[:, l+1, :] / np.linalg.norm(z[:, l+1, :], axis=1)[:, np.newaxis]

movie = False
color = '#3658bf'
now = datetime.now() 
if d == 2:
    dir_path = './circle/beta=%s' % beta 
else:
    dir_path = './sphere/beta=%s' % beta
dt_string = now.strftime("%H-%M-%S")
filename = dt_string + "movie.gif"
base_filename = dt_string
        
if not os.path.exists(dir_path):
     os.makedirs(dir_path)

x_min, x_max = z[:, :, 0].min(), z[:, :, 0].max()
if d>1:
    y_min, y_max = z[:, :, 1].min(), z[:, :, 1].max()
    if d == 3:
        z_min, z_max = z[:, :, 2].min(), z[:, :, 2].max()

margin = 0.1
x_range = x_max - x_min
x_min -= margin * x_range
x_max += margin * x_range

if d>1:
    y_range = y_max - y_min
    y_min -= margin * y_range
    y_max += margin * y_range
    if d == 3:
        z_range = z_max - z_min
        z_min -= margin * z_range
        z_max += margin * z_range
            
rc("text", usetex = True)
font = {'size'   : 18}
rc('font', **font)
        
interp_x = []
interp_y = []
interp_z = []

for i in range(n):
    interp_x.append(interp1d(integration_time, z[i, :, 0], 
                             kind='cubic', 
                             fill_value='extrapolate'))
    if d>1:
        interp_y.append(interp1d(integration_time, z[i, :, 1], 
                                 kind='cubic', 
                                 fill_value='extrapolate'))
        if d==3:
            interp_z.append(interp1d(integration_time, z[i, :, 2], 
                                     kind='cubic', 
                                     fill_value='extrapolate'))

for t in trange(num_steps):
    if d == 2:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1)
        label_size = 16
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.title(r'$t={t}$'.format(t=str(round(t*dt,2))), fontsize=16)
            
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        
        ax.set_xticks([])
        ax.set_yticks([])

        # To remove the axis spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        plt.scatter([x(integration_time)[t] for x in interp_x], 
                    [y(integration_time)[t] for y in interp_y], 
                    c=color, 
                    alpha=1, 
                    marker = 'o', 
                    linewidth=0.75, 
                    edgecolors='black', 
                    zorder=3)
        
        plt.scatter([x(integration_time)[0] for x in interp_x], 
                    [y(integration_time)[0] for y in interp_y], 
                    c='white', 
                    alpha=0.1, 
                    marker = '.', 
                    linewidth=0.3, 
                    edgecolors='black', 
                    zorder=3)
        
        if t > 0:
            for i in range(n):
                x_traj = interp_x[i](integration_time)[:t+1]
                y_traj = interp_y[i](integration_time)[:t+1]
                plt.plot(x_traj, 
                         y_traj, 
                         c=color, 
                         alpha=1, 
                         linewidth = 0.25, 
                         linestyle = 'dashed',
                         zorder=1)
        
        plt.savefig(os.path.join(dir_path, base_filename + "{}.pdf".format(t)), 
                    format='pdf', 
                    bbox_inches='tight')


    elif d == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        label_size = 16
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size
        
        plt.title(r'$t={t}$'.format(t=str(round(t*dt,2))), fontsize=16)
            
        ax.scatter([x(integration_time)[t] for x in interp_x], 
                    [y(integration_time)[t] for y in interp_y],
                    [z(integration_time)[t] for z in interp_z],
                    c=color, 
                    alpha=1, 
                    marker = 'o', 
                    linewidth=0.75, 
                    edgecolors='black')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        
        ax.axis('off')
        
        ax.scatter(z0[:, 0], z0[:, 1], z0[:, 2], 
                    c='lightgray', 
                    alpha=0, 
                    marker = '.', 
                    linewidth=0.0, 
                    edgecolors='lightgray', 
                    zorder=3)
        
        ax.scatter([x(integration_time)[0] for x in interp_x], 
                    [y(integration_time)[0] for y in interp_y],
                    [z(integration_time)[0] for z in interp_z], 
                    c='white', 
                    alpha=0.1, 
                    marker = '.', 
                    linewidth=0.3, 
                    edgecolors='black', 
                    zorder=3)
        
        if t > 0:
            for i in range(n):
                
                ax.scatter(z0[:, 0], z0[:, 1], z0[:, 2],
                            c='lightgray', 
                            alpha=0, 
                            marker = '.', 
                            linewidth=0.0, 
                            edgecolors='lightgray', 
                            zorder=3)
                
                x_traj = interp_x[i](integration_time)[:t+1]
                y_traj = interp_y[i](integration_time)[:t+1]
                z_traj = interp_z[i](integration_time)[:t+1]
                ax.plot(x_traj, 
                        y_traj, 
                        z_traj, 
                        c=color, 
                        alpha=0.75, 
                        linestyle = 'dashed',
                        linewidth = 0.25)
        
        ax.set_xlim3d(x_min, x_max)
        ax.set_ylim3d(y_min, y_max)
        #ax.set_zlim3d(z_min, z_max)
        
        #ax.view_init(elev=10)
        #ax.view_init(elev=10, azim=(45+t/(num_steps-1)*360)%360)
        
        default_azim = 30  

        # the range of rotation around the default_azim
        rotation_range = 30
        total_frames = num_steps
        
        angle = default_azim + rotation_range * (t/total_frames - 0.5)  # will go from -15 to +15 degrees
        ax.view_init(elev=10, azim=angle)
    
        ax.grid(False)
        plt.locator_params(nbins=4)
        
        plt.savefig(os.path.join(dir_path, base_filename + "{}.png".format(t)), 
                    format='png', 
                    bbox_inches='tight',
                    dpi=500)
     
    if movie:           
        plt.savefig(os.path.join(dir_path, base_filename + "{}.png".format(t)),
                    format='png', dpi=250, bbox_inches='tight')
    plt.clf()
    plt.close()

# if movie:
#     imgs = []
#     for i in trange(num_steps):
#         img_file = base_filename + "{}.png".format(i)
#         imgs.append(imageio.imread(img_file))
#         os.remove(img_file) 
#     imageio.mimwrite(filename, imgs)
