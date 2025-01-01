# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Define modern colors for publication
modern_palette = {
    "Male": "#4B8BBE",  # Modern blue
    "Female": "#FF8C42"  # Soft orange
}

# Set file path for data
data_path = "C:/Users/massa/Desktop/MiniProjet/projet passec/Data/PASEC2019_GRADE6_TREAT.dta"

# Load the data
data = pd.read_stata(data_path, convert_categoricals=False)

# Replace numeric values in 'sexe' with descriptive labels
data['sexe'] = data['sexe'].map({1.0: 'Male', 2.0: 'Female'})

# Remove rows with missing 'sexe' values
data = data.dropna(subset=['sexe'])

# Ensure 'PAYS' and 'sexe' are categorical
data['PAYS'] = data['PAYS'].astype('category')
data['sexe'] = data['sexe'].astype('category')

# Define save path for results
save_path = "C:/Users/massa/Desktop/MiniProjet/projet passec/Figure python/"
os.makedirs(save_path, exist_ok=True)

# Function to resize and save images
def resize_and_save(fig, filename, max_size=(3000, 3000)):
    temp_path = save_path + filename
    fig.savefig(temp_path, dpi=400, bbox_inches='tight')  # Save at high quality
    # Open and resize
    image = Image.open(temp_path)
    image_resized = image.resize(max_size, Image.Resampling.LANCZOS)
    image_resized.save(temp_path, optimize=True)
    print(f"Image saved and resized to {max_size}: {temp_path}")

# Custom function to clean the legend
def clean_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["Male" if "Male" in label else "Female" for label in labels]
    ax.legend(handles, new_labels, title="Gender", fontsize=10, title_fontsize=10)

# Plot 1: Overall Distribution of Math Scores by Gender
fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=data, x="MATHS_PV5", hue="sexe", fill=True, palette=modern_palette, alpha=0.5)
plt.title("Overall Distribution of Mathematics Scores by Gender", fontsize=16, fontweight="bold")
plt.xlabel("Mathematics Scores", fontsize=12)
plt.ylabel("")
clean_legend(ax)
plt.tight_layout()
resize_and_save(fig, "Overall_Distribution_MATHS_PV5.png")

# Plot 2: Overall Distribution of Reading Scores by Gender
fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=data, x="LECT_PV5", hue="sexe", fill=True, palette=modern_palette, alpha=0.5)
plt.title("Overall Distribution of Reading Scores by Gender", fontsize=16, fontweight="bold")
plt.xlabel("Reading Scores", fontsize=12)
plt.ylabel("")
clean_legend(ax)
plt.tight_layout()
resize_and_save(fig, "Overall_Distribution_LECT_PV5.png")

# Plot 3: Distribution by Country and Gender (Faceted Grid for Math Scores)
g = sns.FacetGrid(data, col="PAYS", hue="sexe", palette=modern_palette, col_wrap=4, height=4, sharey=False)
g.map(sns.kdeplot, "MATHS_PV5", common_norm=False, fill=True, alpha=0.5)

# Remove Y-axis label, X-axis label, and Y-axis ticks
for ax, title in zip(g.axes.flat, g.col_names):
    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks([])

g.add_legend(title="Gender")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Distribution of Mathematics Test Scores by Country and Gender", fontsize=16, fontweight="bold")
resize_and_save(g.fig, "Facet_Distribution_MATHS_PV5.png")

# Plot 4: Distribution by Country and Gender (Faceted Grid for Reading Scores)
g = sns.FacetGrid(data, col="PAYS", hue="sexe", palette=modern_palette, col_wrap=4, height=4, sharey=False)
g.map(sns.kdeplot, "LECT_PV5", common_norm=False, fill=True, alpha=0.5)

# Remove Y-axis label, X-axis label, and Y-axis ticks
for ax, title in zip(g.axes.flat, g.col_names):
    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks([])

g.add_legend(title="Gender")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Distribution of Reading Test Scores by Country and Gender", fontsize=16, fontweight="bold")
resize_and_save(g.fig, "Facet_Distribution_LECT_PV5.png")

# Plot 5: Box Plot of Mathematics Scores by Country and Gender
fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=data, x="PAYS", y="MATHS_PV5", hue="sexe", palette=modern_palette)
plt.title("Box Plot of Mathematics Scores by Country and Gender", fontsize=16, fontweight="bold")
plt.xlabel("Country", fontsize=12)
plt.ylabel("Mathematics Scores", fontsize=12)
clean_legend(ax)
plt.xticks(rotation=45)  # Rotate country names at 45 degrees
plt.tight_layout()
fig.savefig(save_path + "BoxPlot_MATHS_PV5.png", dpi=400, bbox_inches='tight')  # Save without resizing
print(f"Box Plot for Mathematics Scores saved: {save_path}BoxPlot_MATHS_PV5.png")

# Plot 6: Box Plot of Reading Scores by Country and Gender
fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=data, x="PAYS", y="LECT_PV5", hue="sexe", palette=modern_palette)
plt.title("Box Plot of Reading Scores by Country and Gender", fontsize=16, fontweight="bold")
plt.xlabel("Country", fontsize=12)
plt.ylabel("Reading Scores", fontsize=12)
clean_legend(ax)
plt.xticks(rotation=45)  # Rotate country names at 45 degrees
plt.tight_layout()
fig.savefig(save_path + "BoxPlot_LECT_PV5.png", dpi=400, bbox_inches='tight')  # Save without resizing
print(f"Box Plot for Reading Scores saved: {save_path}BoxPlot_LECT_PV5.png")
