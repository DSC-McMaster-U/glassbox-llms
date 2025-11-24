import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# optional, can use umap or t-sne instead (this one uses t-sne)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

######################################

# PLOT 1
# This is an interactive scatter plot that can show multiple data sets.
# Could be useful after projecting higher-dimensional data to 2D
# Can cluster together related concepts/show connections
# Can identify outliers

np.random.seed(42)

# some data points for demonstration
x_data_group1 = np.random.rand(50) * 10
y_data_group1 = np.random.rand(50) * 5
labels_group1 = [f"Group A - Point {i}" for i in range(50)]

# this data group has an example shift
x_data_group2 = np.random.rand(50) * 10 + 2
y_data_group2 = np.random.rand(50) * 5 + 3
labels_group2 = [f"Group B - Point {i}" for i in range(50)]

fig1 = go.Figure()

fig1.add_trace(
    go.Scatter(
        x=x_data_group1,
        y=y_data_group1,
        mode="markers",
        name="Group A",  # legend name
        hoverinfo="text",
        hovertext=labels_group1,  # you can choose names on hover defined programmaticaly
        marker=dict(
            size=10,
            opacity=0.7,
            color="blue",
        ),
    )
)

fig1.add_trace(
    go.Scatter(
        x=x_data_group2,
        y=y_data_group2,
        mode="markers",
        name="Group B",
        hoverinfo="text",
        hovertext=labels_group2,
        marker=dict(
            size=10,
            opacity=0.7,
            color="red",
        ),
    )
)

fig1.update_layout(
    title="Interactive Scatter Plot with Multiple Traces",
    xaxis_title="Feature 1",
    yaxis_title="Feature 2",
    hovermode="closest",
)
fig1.show()

######################################

# PLOT 2
# This is like the last one, but with 3D
# Plotly lets you rotate and zoom it
# Can cluster together related concepts/show connections
# Can identify outliers

# here's some example data in a higher dimension (10) projected down using t-SNE
num = 100
words = [f"word_{i}" for i in range(num)]
embeddings = np.random.rand(num, 10) * 10
tsne = TSNE(
    n_components=3,
    random_state=42,
    perplexity=10,
)
embeddings_3d_tsne = tsne.fit_transform(embeddings)

# ignore the error you may get here, your type checker is wrong
df_tsne = pd.DataFrame(
    embeddings_3d_tsne,
    columns=[
        "dim1",
        "dim2",
        "dim3",
    ],
)
df_tsne["word"] = words
df_tsne["category"] = np.random.choice(
    [
        "topicA",
        "topicB",
        "topicC",
    ],
    num,
)

fig2 = px.scatter_3d(
    df_tsne,
    x="dim1",
    y="dim2",
    z="dim3",
    color="category",  # we can color each point based on an attribute
    hover_name="word",
    title="3D Scatter Visualization",
)
fig2.update_layout(
    scene_camera=dict(
        eye=dict(
            x=1.5,
            y=1.5,
            z=1.5,
        )
    )
)  # this is the default camera angle
fig2.show()

######################################

# PLOT 3
# This is a heatmap.
# Helpful when dealing with matrix-like data (eg. attention weights, correlation matrices, etc)
# Dark = stronger connection, vice versa for light colors

# example attention matrix (e.g. output from a single attention head)
seq_len = 10
tokens = [f"Token_{i}" for i in range(seq_len)]
weights = np.random.rand(
    seq_len,
    seq_len,
)
np.fill_diagonal(
    weights,
    1.0,
)  # token[i] will obviously be very similar to itself
weights = (
    weights
    / weights.sum(axis=1)[
        :,
        None,
    ]
)  # normalize each row

fig3 = go.Figure(
    data=go.Heatmap(
        z=weights,  # each value in the matrix = 1 cell
        x=tokens,
        y=tokens,
        colorscale="Viridis",  # this is just a color theme
        colorbar=dict(title="Attention Weight"),
    )
)

fig3.update_layout(
    title="Attention Heatmap",
    xaxis_title="Q Token",
    yaxis_title="K Token",
)
fig3.show()

######################################

# PLOT 4
# It's a bar chart.
# Pretty self-explanatory, can be used to display discrete values, so on

# some common example tokens
top_k_tokens = [
    "the",
    "a",
    "is",
    "of",
    "in",
    "and",
    "to",
    "for",
]
probabilities = np.random.rand(len(top_k_tokens))
probabilities = probabilities / probabilities.sum()  # normalize
probabilities = np.sort(probabilities)[::-1]  # optional but makes for better visualization

fig4 = go.Figure(
    data=[
        go.Bar(
            x=top_k_tokens,
            y=probabilities,
            marker_color="skyblue",
        )
    ]
)

fig4.update_layout(
    title="Top Next Token Predictions",
    xaxis_title="Token",
    yaxis_title="Probability",
    yaxis_range=[
        0,
        1,
    ],
)
fig4.show()

######################################

# PLOT 5
# This is a network graph.
# I'm not actually sure how useful this is, but it explicitly shows connections between nodes which is nice.
# Can help with co-occurence? I'm sure this would be more useful when used with actual data

# nodes represent words and edges represent connections
nodes = [
    "cat",
    "dog",
    "pet",
    "animal",
    "walk",
    "food",
    "play",
    "house",
]
edges = [
    ("cat", "pet"),
    ("dog", "pet"),
    ("pet", "animal"),
    ("dog", "walk"),
    ("cat", "food"),
    ("dog", "food"),
    ("walk", "play"),
    ("pet", "house"),
    ("cat", "play"),
]

# this part can be done via algorithm
pos = {
    "cat": (0, 1),
    "dog": (1, 1),
    "pet": (0.5, 2),
    "animal": (0.5, 3),
    "walk": (2, 1),
    "food": (0.5, 0),
    "play": (1.5, 2),
    "house": (-0.5, 1),
}

edge_x = []
edge_y = []
for edge in edges:
    # gets start/end points and connects them
    (x0, y0) = pos[edge[0]]
    (x1, y1) = pos[edge[1]]
    edge_x.extend([x0, x1, None])  # the None signals end of a line
    edge_y.extend([y0, y1, None])

node_x = [pos[node][0] for node in nodes]
node_y = [pos[node][1] for node in nodes]

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=0.5, color="#888"),
    hoverinfo="none",
    mode="lines",
)  # note it's the same structure as a Scatter but with mode="lines"

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    hoverinfo="text",
    text=nodes,
    textposition="bottom center",
    marker=dict(
        symbol="circle",
        size=10,
        color="#FECF5F",
        line=dict(color="rgb(200,200,200)", width=0.5),
    ),
    textfont=dict(size=10),
)

fig5 = go.Figure(data=[edge_trace, node_trace])

fig5.update_layout(
    title="Word Network Graph",
    showlegend=False,
    hovermode="closest",
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=600,
    dragmode="pan",  # or "select" for box/lasso selection
)
fig5.show()

# --- 6. Faceting / Subplots (Comparing different LLM variants or layers) ---
print("--- 6. Faceting / Subplots: Comparing and Contrasting Across Models or Layers ---")
print(
    "Plotly Express makes it easy to create multiple plots in a grid (facets), which is incredibly useful for comparing different LLM configurations, layers, or attention heads.\n"
)

######################################

# PLOT 6
# You can do this with most chart types, but plotly lets you put multiple plots (facets) in a grid
# Here is an example with a line graph

# example data for comparing two 'models'
df_compare = pd.DataFrame(
    {
        "Metric": np.random.rand(20) * 10,
        "Model": ["Model A"] * 10 + ["Model B"] * 10,
        "Epoch": list(range(10)) * 2,
        "Category": np.random.choice(
            [
                "C1",
                "C2",
            ],
            20,
        ),
    }
)

fig6 = px.line(
    df_compare,
    x="Epoch",
    y="Metric",
    color="Category",
    line_group="Category",
    facet_col="Model",  # creates separate columns for each 'Model'
    # facet_row='Category', # could also use facet_row
    title="Metric Comparison On Two Models",
)
fig6.show()

######################################

# Just a demonstration of how you can customize the infobox on hover

# we're just going to use the t-sne data
fig7 = go.Figure(
    data=[
        go.Scatter3d(
            x=df_tsne["dim1"],
            y=df_tsne["dim2"],
            z=df_tsne["dim3"],
            mode="markers",
            marker=dict(
                size=8,
                color=df_tsne["dim3"],  # color by one of the t-SNE components
                colorscale="Plasma",
                colorbar=dict(title="dim3 Value"),
            ),
            # custom hover template
            hovertemplate=(
                "<b>Word:</b> %{customdata[0]}<br>"
                + "<b>Category:</b> %{customdata[1]}<br>"
                + "<b>Data for dim1:</b> %{x:.2f}<br>"
                + "<b>Data for dim2:</b> %{y:.2f}<br>"
                + "<b>Data for dim3:</b> %{z:.2f}<br>"
                + "Some other mysterious other text you want here <extra></extra>"  # <extra></extra> removes default box
            ),
            customdata=df_tsne[
                [
                    "word",
                    "category",
                ]
            ],  # you need to define what data you want available in the hovertemplate
        )
    ]
)

fig7.update_layout(
    title="3D t-SNE 2: Bonus Hover Content",
    scene=dict(
        xaxis_title="dim1",
        yaxis_title="dim2",
        zaxis_title="dim3",
    ),
)
fig7.show()
