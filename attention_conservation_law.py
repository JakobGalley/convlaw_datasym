import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines


### Model ###

def lightning_attention(params, X):
    Q, K, V = params

    Z = np.einsum("sni,snj->sij", X, X)          # X^T X, shape (N,d,d)
    A = np.einsum("sni,ir->snr", X, Q)          # XQ, shape (N,n,r)
    C = np.einsum("ir,sij,jm->srm", K, Z, V)    # K^T Z V, shape (N,r,m)
    F = np.einsum("snr,srm->snm", A, C)         # shape (N,n,m)

    return F


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

np.random.seed(13)

# Dimensions
d = 8
n = 5
r = 1
m = 1

num_base = 10

num_haar_pairs = 2048

steps = 150
lr = 1.0e-5

data_scale = 0.80
teacher_scale = 1.50
init_scale = 0.40

### Creation of training data ###

X_base = data_scale * np.random.randn(num_base, n, d)


teacher_params = (
    teacher_scale * np.random.randn(d, r),
    teacher_scale * np.random.randn(d, r),
    teacher_scale * np.random.randn(d, m),
)

Y_base = lightning_attention(teacher_params, X_base)


def random_orthogonal(d):
    """
    Sample a Haar-distributed element of O(d) using QR decomposition.
    """
    A = np.random.randn(d, d)
    Q, R = np.linalg.qr(A)

    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    Q = Q * signs

    return Q


G_samples = []

for _ in range(num_haar_pairs):
    g = random_orthogonal(d)
    G_samples.append(g)
    G_samples.append(-g)

G_samples = np.stack(G_samples, axis=0)
num_haar_samples = G_samples.shape[0]

print("Number of Haar samples:", num_haar_samples)
print("Number of augmented data points:", num_base * num_haar_samples)


X_aug = np.einsum("sni,kij->sknj", X_base, G_samples).reshape(
    num_base * num_haar_samples, n, d
)

# Labels are invariant under the right O(d)-action.
Y_aug = np.repeat(Y_base, num_haar_samples, axis=0)

### Setup gradient descent and conservation laws ###

def loss_and_grad(params, X, Y):
    Q, K, V = params
    N = X.shape[0]

    # Forward pass.
    Z = np.einsum("sni,snj->sij", X, X)
    A = np.einsum("sni,ir->snr", X, Q)
    C = np.einsum("ir,sij,jm->srm", K, Z, V)
    F = np.einsum("snr,srm->snm", A, C)

    err = F - Y
    loss = np.mean(np.sum(err * err, axis=(1, 2)))

    # Gradient of mean squared error.
    H = (2.0 / N) * err

    # Since F = A C.
    grad_Q = np.einsum("sni,snm,srm->ir", X, H, C)

    # Gradient with respect to C.
    grad_C = np.einsum("snr,snm->srm", A, H)

    # Since C = K^T Z V.
    grad_K = np.einsum("sij,jm,srm->ir", Z, V, grad_C)
    grad_V = np.einsum("sij,jr,srm->im", Z, K, grad_C)

    return loss, (grad_Q, grad_K, grad_V)


def projector_from_params(params):
    """
    Orthogonal projector onto range(P), where

        P = [Q K V].

    Since r=m=1, P has three columns. We initialize P with full column rank,
    and use the top three left singular vectors to represent its column space.
    """
    Q, K, V = params
    P = np.concatenate([Q, K, V], axis=1)

    U, s, _ = np.linalg.svd(P, full_matrices=False)

    # Fixed rank 3 because P = [Q K V] has three columns and is initialized full rank.
    U = U[:, :3]

    return U @ U.T


def column_space_distance(params, Pi0):
    """
    Frobenius distance between the current column-space projector and
    the initial column-space projector.
    """
    Pit = projector_from_params(params)
    return np.linalg.norm(Pit - Pi0, ord="fro")


def grad_standard(params):
    return loss_and_grad(params, X_base, Y_base)


def grad_augmented(params):
    return loss_and_grad(params, X_aug, Y_aug)


def gradient_descent(grad_fn, params_init):
    params = tuple(p.copy() for p in params_init)
    Pi0 = projector_from_params(params)

    distances = []
    losses = []

    for _ in range(steps + 1):
        distances.append(column_space_distance(params, Pi0))

        loss, grads = grad_fn(params)
        losses.append(loss)

        params = tuple(p - lr * g for p, g in zip(params, grads))

    return np.array(distances), np.array(losses)


def random_initialization(min_singular_value=0.08):
   

    while True:
        Q0 = init_scale * np.random.randn(d, r)
        K0 = init_scale * np.random.randn(d, r)
        V0 = init_scale * np.random.randn(d, m)

        P0 = np.concatenate([Q0, K0, V0], axis=1)
        s = np.linalg.svd(P0, compute_uv=False)

        if s[-1] > min_singular_value:
            return Q0, K0, V0


initializations = [random_initialization() for _ in range(6)]

dist_standard = []
dist_augmented = []

for init in initializations:
    d_std, loss_std = gradient_descent(grad_standard, init)
    d_aug, loss_aug = gradient_descent(grad_augmented, init)

    dist_standard.append(d_std)
    dist_augmented.append(d_aug)

dist_standard = np.array(dist_standard)
dist_augmented = np.array(dist_augmented)

print("\nFinal standard column-space distances:")
print(dist_standard[:, -1])

print("\nFinal augmented column-space distances:")
print(dist_augmented[:, -1])

print("\nMax standard distance:", np.max(dist_standard))
print("Max augmented distance:", np.max(dist_augmented))

### Plotting ###

fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

# One color for each dynamics type.
std_color = "#4C72B0"
aug_color = "#DD8452"

# Floor values for log scale.
dist_standard_plot = np.maximum(dist_standard, 1e-16)
dist_augmented_plot = np.maximum(dist_augmented, 1e-16)

# Mean over all runs.
mean_standard = np.mean(dist_standard_plot, axis=0)
mean_augmented = np.mean(dist_augmented_plot, axis=0)


for dev_std in dist_standard_plot:
    ax.plot(
        dev_std,
        linestyle='--',
        color=std_color,
        alpha=0.22,
        linewidth=1.0,
        zorder=1
    )

for dev_aug in dist_augmented_plot:
    ax.plot(
        dev_aug,
        linestyle='-',
        color=aug_color,
        alpha=0.22,
        linewidth=1.0,
        zorder=1
    )


ax.plot(
    mean_standard,
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
    mean_augmented,
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

# Log scale.
ax.set_yscale('log')
ax.set_ylim(1e-6, 3.0)
ax.set_xlim(0, steps)

# Labels.
ax.set_xlabel(r'Gradient Flow Step ($t$)', labelpad=10)
ax.set_ylabel(
    r'$\|\Pi_{\mathrm{range}(P_t)}-\Pi_{\mathrm{range}(P_0)}\|_F$',
    labelpad=10
)


ax.grid(True, which="major", linestyle='-', alpha=0.1, color='black')
ax.grid(False, which="minor")

# Legend proxies.
std_runs_proxy = mlines.Line2D(
    [],
    [],
    color=std_color,
    linestyle='--',
    linewidth=1.0,
    alpha=0.25,
    label=r'Standard Dynamics, individual runs'
)

std_mean_proxy = mlines.Line2D(
    [],
    [],
    color=std_color,
    linestyle='--',
    linewidth=2.4,
    label=r'Standard Dynamics, mean'
)

aug_runs_proxy = mlines.Line2D(
    [],
    [],
    color=aug_color,
    linestyle='-',
    linewidth=1.0,
    alpha=0.25,
    label=r'Augmented Dynamics, individual runs'
)

aug_mean_proxy = mlines.Line2D(
    [],
    [],
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
