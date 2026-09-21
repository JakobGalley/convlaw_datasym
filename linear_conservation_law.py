import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines

### Plot parameters ###

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
    "axes.labelsize": 15,
    "font.size": 13,
    "legend.fontsize": 12,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False
})


### Other parameters ###

np.random.seed(42)

n_base = 4

### Creation of training data ###

X_base = np.random.randn(n_base, 3) * 2.0
Y_base = np.random.randn(n_base, 1)


X_aug = []
Y_aug = []
for i in range(n_base):
    x = X_base[i]
    y = Y_base[i]

    X_aug.extend([
        x,
        np.array([x[1], x[2], x[0]]),
        np.array([x[2], x[0], x[1]])
    ])
    Y_aug.extend([y, y, y])

X_base = np.array(X_base)
Y_base = np.array(Y_base)
X_aug = np.array(X_aug)
Y_aug = np.array(Y_aug)

### Setup gradient descent and conservation laws ###

def grad_standard(W):
    err = (X_base @ W) - Y_base.flatten()
    return np.mean(2.0 * err[:, None] * X_base, axis=0)

def grad_augmented(W):
    err = (X_aug @ W) - Y_aug.flatten()
    return np.mean(2.0 * err[:, None] * X_aug, axis=0)

def gradient_descent(grad_fn, W_init, lr=0.015, steps=150):
    trajectory = [W_init]
    W = np.copy(W_init)

    for _ in range(steps):
        W = W - lr * grad_fn(W)
        trajectory.append(np.copy(W))

    return np.array(trajectory)

def compute_integral(W):
    num = W[0] - W[1]
    den = W[0] + W[1] - 2.0 * W[2]
    # return den/num if np.abs(den) > 1e-8 else np.nan
    return num / den if np.abs(den) > 1e-8 else np.nan

def compute_deviation(traj, floor=1e-16):
    I_init = compute_integral(traj[0])
    dev = np.array([
        np.abs(compute_integral(w) - I_init)
        for w in traj
    ])

    # Floor values for log scale
    dev = np.maximum(dev, floor)
    return dev


initializations = [
    np.array([4.0, 3.0, -1.0]),
    np.array([2.5, 4.5, 1.0]),
    np.array([-3.0, 4.0, -2.5]),
    np.array([-4.5, 2.0, 3.0]),
    np.array([-4.0, -3.5, 2.0]),
    np.array([3.5, -4.0, -1.5])
]

trajectories_std = [
    gradient_descent(grad_standard, init)
    for init in initializations
]

trajectories_aug = [
    gradient_descent(grad_augmented, init)
    for init in initializations
]


devs_std = np.array([
    compute_deviation(traj)
    for traj in trajectories_std
])

devs_aug = np.array([
    compute_deviation(traj)
    for traj in trajectories_aug
])


mean_std = np.nanmean(devs_std, axis=0)
mean_aug = np.nanmean(devs_aug, axis=0)

### Plotting ###

fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

# One color for each dynamics type
std_color = "#4C72B0"
aug_color = "#DD8452"


for dev_std in devs_std:
    ax.plot(
        dev_std,
        linestyle='--',
        color=std_color,
        alpha=0.22,
        linewidth=1.0,
        zorder=1
    )

for dev_aug in devs_aug:
    ax.plot(
        dev_aug,
        linestyle='-',
        color=aug_color,
        alpha=0.22,
        linewidth=1.0,
        zorder=1
    )


ax.plot(
    mean_std,
    linestyle='--',
    color=std_color,
    alpha=1.0,
    linewidth=2.4,
    zorder=3,
    path_effects=[
        pe.Stroke(linewidth=3.8, foreground='white'),
        pe.Normal()
    ]
)

ax.plot(
    mean_aug,
    linestyle='-',
    color=aug_color,
    alpha=1.0,
    linewidth=2.6,
    zorder=4,
    path_effects=[
        pe.Stroke(linewidth=4.0, foreground='white'),
        pe.Normal()
    ]
)

# Log scale
ax.set_yscale('log')
ax.set_ylim(1e-17, 1e3)
ax.set_xlim(0, 150)

# Labels
ax.set_xlabel(r'Gradient Flow Step ($t$)', labelpad=10)
ax.set_ylabel(r'$|I(W_t) - I(W_0)|$', labelpad=10)

# Grid
ax.grid(True, which="both", linestyle='-', alpha=0.1, color='black')

# Legend
std_runs_proxy = mlines.Line2D(
    [], [],
    color=std_color,
    linestyle='--',
    linewidth=1.0,
    alpha=0.25,
    label=r'Standard Dynamics, individual runs'
)

std_mean_proxy = mlines.Line2D(
    [], [],
    color=std_color,
    linestyle='--',
    linewidth=2.4,
    label=r'Standard Dynamics, mean'
)

aug_runs_proxy = mlines.Line2D(
    [], [],
    color=aug_color,
    linestyle='-',
    linewidth=1.0,
    alpha=0.25,
    label=r'Augmented Dynamics, individual runs'
)

aug_mean_proxy = mlines.Line2D(
    [], [],
    color=aug_color,
    linestyle='-',
    linewidth=2.6,
    label=r'Augmented Dynamics, mean'
)

ax.legend(
    handles=[
        std_runs_proxy,
        std_mean_proxy,
        aug_runs_proxy,
        aug_mean_proxy
    ],
    loc='lower center',
    bbox_to_anchor=(0.5, 1.05),
    ncol=2,
    frameon=False,
    borderpad=0,
    columnspacing=1.4,
    handlelength=2.8
)

plt.tight_layout()
plt.show()
