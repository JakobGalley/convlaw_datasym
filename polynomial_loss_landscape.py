import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

### Model  ###

def model(z):
    return 0.012 * z**6 - 0.22 * z**4 + z**2

### Creation of training data ###

X_raw = np.array([
    [ 1.00,  0.20],
    [-0.38,  0.92],
    [ 0.78, 0.62],
    [-0.90, -0.18],
], dtype=float)

y_data = np.array([0.85, 0.35, 1.05, 0.62], dtype=float)



directions = X_raw / np.linalg.norm(X_raw, axis=1, keepdims=True)

data_radii = np.array([0.92, 0.48, 1.04, 1.10])
X_data = data_radii[:, None] * directions



num_group_elements = 18
angles = np.linspace(0, 2 * np.pi, num_group_elements, endpoint=False)

G = np.stack([
    np.array([
        [np.cos(a), -np.sin(a)],
        [np.sin(a),  np.cos(a)]
    ])
    for a in angles
], axis=0)



n_grid = 260
lim = 2.4

### Calculate loss landscape ###

p1 = np.linspace(-lim, lim, n_grid)
p2 = np.linspace(-lim, lim, n_grid)

P1, P2 = np.meshgrid(p1, p2)
P = np.stack([P1.ravel(), P2.ravel()], axis=1)



z_plain = X_data @ P.T
pred_plain = model(z_plain)

loss_plain = np.mean((pred_plain - y_data[:, None])**2, axis=0)
Z_plain = loss_plain.reshape(P1.shape)



X_aug = np.einsum("gij,sj->gsi", G, X_data)
z_aug = X_aug @ P.T
# z_aug = np.einsum("gsi,ki->gsk", X_aug, P)
pred_aug = model(z_aug)
loss_aug = np.mean((pred_aug - y_data[None, :, None])**2, axis=(0, 1))
Z_aug = loss_aug.reshape(P1.shape)



Z_plain_plot = Z_plain
Z_aug_plot = Z_aug


joint_upper_plot = np.percentile(
    np.concatenate([Z_plain.ravel(), Z_aug.ravel()]), 
    92
)



X_aug_plot = np.einsum("gij,sj->sgi", G, X_data)
X_aug_flat = X_aug_plot.reshape(-1, 2)

y_aug_flat = np.repeat(y_data, num_group_elements)

### Plotting ###

plain_scale = [
    [0.00, "#2b183f"],
    [0.18, "#58305f"],
    [0.38, "#9a4f53"],
    [0.62, "#d07c46"],
    [0.82, "#efb05f"],
    [1.00, "#f6df93"],
]

aug_scale = [
    [0.00, "#102a43"],
    [0.18, "#15566b"],
    [0.38, "#1f8a8a"],
    [0.62, "#52b69a"],
    [0.82, "#b5e48c"],
    [1.00, "#f9dc5c"],
]

label_scale = [
    [0.00, "#2b183f"],
    [0.18, "#58305f"],
    [0.38, "#9a4f53"],
    [0.62, "#d07c46"],
    [0.82, "#efb05f"],
    [1.00, "#f6df93"],
]

label_min = float(np.min(y_data))
label_max = float(np.max(y_data))



fig = make_subplots(
    rows=2,
    cols=2,
    specs=[
        [{"type": "surface"}, {"type": "surface"}],
        [{"type": "xy"},      {"type": "xy"}],
    ],
    horizontal_spacing=0.015,
    vertical_spacing=0.030,
    row_heights=[0.64, 0.36],
)


surface_style = dict(
    showscale=False,
    opacity=1.0,
    lighting=dict(
        ambient=0.52,
        diffuse=0.92,
        specular=0.58,
        roughness=0.24,
        fresnel=0.12,
    ),
    lightposition=dict(x=120, y=140, z=220),
)



fig.add_trace(
    go.Surface(
        x=P1,
        y=P2,
        z=Z_plain_plot,
        colorscale=plain_scale,
        cmin=0.0,
        cmax=joint_upper_plot,
        **surface_style,
    ),
    row=1,
    col=1,
)



fig.add_trace(
    go.Surface(
        x=P1,
        y=P2,
        z=Z_aug_plot,
        colorscale=aug_scale,
        cmin=0.0,
        cmax=joint_upper_plot,
        **surface_style,
    ),
    row=1,
    col=2,
)



plane_lim = 1.32
theta = np.linspace(0, 2 * np.pi, 5000)


def add_input_plane(fig, row, col, show_orbit_guides):
    fig.add_trace(
        go.Scatter(
            x=[-plane_lim, plane_lim, plane_lim, -plane_lim, -plane_lim],
            y=[-plane_lim, -plane_lim, plane_lim, plane_lim, -plane_lim],
            mode="lines",
            fill="toself",
            fillcolor="rgba(248,248,250,0.86)",
            line=dict(color="rgba(90,90,90,0.20)", width=1.0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    
    fig.add_trace(
        go.Scatter(
            x=[-plane_lim, plane_lim],
            y=[0, 0],
            mode="lines",
            line=dict(color="rgba(60,60,60,0.30)", width=1.15),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[-plane_lim, plane_lim],
            mode="lines",
            line=dict(color="rgba(60,60,60,0.30)", width=1.15),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

   
    if show_orbit_guides:
        for r in data_radii:
            fig.add_trace(
                go.Scatter(
                    x=r * np.cos(theta),
                    y=r * np.sin(theta),
                    mode="lines",
                    line=dict(
                        color="rgba(70,70,70,0.14)",
                        width=1.05,
                        dash="dot",
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )



add_input_plane(fig, row=2, col=1, show_orbit_guides=False)
add_input_plane(fig, row=2, col=2, show_orbit_guides=True)


fig.add_trace(
    go.Scatter(
        x=X_data[:, 0],
        y=X_data[:, 1],
        mode="markers",
        marker=dict(
            size=25,
            color="rgba(0,0,0,0.14)",
            line=dict(width=0),
        ),
        hoverinfo="skip",
        showlegend=False,
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=X_data[:, 0],
        y=X_data[:, 1],
        mode="markers",
        marker=dict(
            size=15,
            color=y_data,
            colorscale=label_scale,
            cmin=label_min,
            cmax=label_max,
            showscale=False,
            line=dict(color="rgba(255,255,255,0.95)", width=1.15),
        ),
        customdata=y_data[:, None],
        hovertemplate=(
            "x₁ = %{x:.3f}<br>"
            "x₂ = %{y:.3f}<br>"
            "label = %{customdata[0]:.3f}"
            "<extra></extra>"
        ),
        showlegend=False,
    ),
    row=2,
    col=1,
)


fig.add_trace(
    go.Scatter(
        x=X_aug_flat[:, 0],
        y=X_aug_flat[:, 1],
        mode="markers",
        marker=dict(
            size=9,
            color=y_aug_flat,
            colorscale=label_scale,
            cmin=label_min,
            cmax=label_max,
            showscale=True,
            colorbar=dict(
                thickness=11,
                len=0.33,
                x=1.035,
                y=0.18,
                outlinewidth=0,
                tickfont=dict(size=10),
            ),
            line=dict(color="rgba(255,255,255,0.82)", width=0.55),
        ),
        customdata=y_aug_flat[:, None],
        hovertemplate=(
            "x₁ = %{x:.3f}<br>"
            "x₂ = %{y:.3f}<br>"
            "label = %{customdata[0]:.3f}"
            "<extra></extra>"
        ),
        showlegend=False,
    ),
    row=2,
    col=2,
)



axis_style_3d = dict(
    title="",
    showbackground=False,
    showgrid=False,
    zeroline=False,
    showline=True,
    linewidth=2.0,
    linecolor="rgba(70,70,70,0.95)",
    ticks="",
    showticklabels=False,
)

camera = dict(
    eye=dict(x=1.45, y=1.36, z=0.82)
)



axis_style_2d = dict(
    title="",
    showgrid=False,
    zeroline=False,
    showline=False,
    ticks="",
    showticklabels=False,
    range=[-plane_lim, plane_lim],
    fixedrange=True,
)


fig.update_layout(
    width=800,
    height=680,
    margin=dict(l=0, r=18, t=0, b=0),
    paper_bgcolor="white",
    plot_bgcolor="white",

    scene=dict(
        xaxis=axis_style_3d,
        yaxis=axis_style_3d,
        zaxis=axis_style_3d,
        camera=camera,
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.48),
    ),

    scene2=dict(
        xaxis=axis_style_3d,
        yaxis=axis_style_3d,
        zaxis=axis_style_3d,
        camera=camera,
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.48),
    ),
)

fig.update_xaxes(axis_style_2d, row=2, col=1)
fig.update_yaxes(axis_style_2d, row=2, col=1, scaleanchor="x", scaleratio=1)

fig.update_xaxes(axis_style_2d, row=2, col=2)
fig.update_yaxes(axis_style_2d, row=2, col=2, scaleanchor="x2", scaleratio=1)

fig.show()