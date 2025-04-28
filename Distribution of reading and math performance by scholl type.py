# Import des librairies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# Configuration esthétique
sns.set_theme(style="whitegrid")
school_palette = {
    "Public": "#4B8BBE",
    "Private Secular": "#FF8C42",
    "Private Religious": "#9C27B0",
    "Community": "#2E8B57"  # inclure si présent
}

# Chemins
data_path = "C:/Users/massa/Desktop/projet passec/Data/PASEC2019_GRADE6_TREAT.dta"
save_path = "C:/Users/massa/Desktop/projet passec/Figure python/"
os.makedirs(save_path, exist_ok=True)

# Chargement des données (évite le UnicodeWarning)
data = pd.read_stata(data_path, convert_categoricals=False).copy()

# Création de la variable 'school_type'
school_type_map = {
    1: 'Public',
    2: 'Private Secular',
    3: 'Private Religious',
    4: 'Community'  # facultatif selon ta base
}
data['school_type'] = data['qd17'].map(school_type_map)
data = data.dropna(subset=['school_type'])

# Fonction pour sauvegarder et redimensionner les figures
def resize_and_save(fig, filename, max_size=(3000, 3000)):
    temp_path = save_path + filename
    fig.savefig(temp_path, dpi=400, bbox_inches='tight')
    image = Image.open(temp_path)
    image_resized = image.resize(max_size, Image.Resampling.LANCZOS)
    image_resized.save(temp_path, optimize=True)
    print(f"Image enregistrée : {temp_path}")

# ==== DISTRIBUTION GLOBALE ====

# Mathématiques
fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=data, x="MATHS_PV5", hue="school_type", fill=True, alpha=0.4, palette=school_palette)
plt.title("Distribution Globale des Scores en Mathématiques par Type d’École", fontsize=16, fontweight="bold")
plt.xlabel("Scores en Mathématiques", fontsize=12)
plt.ylabel("")
ax.legend(title="Type d'École")
plt.tight_layout()
resize_and_save(fig, "DistributionGlobale_Maths_SchoolType.png")

# Lecture
fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=data, x="LECT_PV5", hue="school_type", fill=True, alpha=0.4, palette=school_palette)
plt.title("Distribution Globale des Scores en Lecture par Type d’École", fontsize=16, fontweight="bold")
plt.xlabel("Scores en Lecture", fontsize=12)
plt.ylabel("")
ax.legend(title="Type d'École")
plt.tight_layout()
resize_and_save(fig, "DistributionGlobale_Lecture_SchoolType.png")

# ==== DISTRIBUTION PAR PAYS ====

# FacetGrid Mathématiques
g = sns.FacetGrid(data, col="PAYS", hue="school_type", col_wrap=4, height=4, sharey=False, palette=school_palette)
g.map(sns.kdeplot, "MATHS_PV5", common_norm=False, fill=True, alpha=0.4)
for ax in g.axes.flat:
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks([])
g.add_legend(title="Type d'École")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Distribution des Scores en Mathématiques par Pays et Type d’École", fontsize=16, fontweight="bold")
resize_and_save(g.fig, "DistributionPays_Maths_SchoolType.png")

# FacetGrid Lecture
g = sns.FacetGrid(data, col="PAYS", hue="school_type", col_wrap=4, height=4, sharey=False, palette=school_palette)
g.map(sns.kdeplot, "LECT_PV5", common_norm=False, fill=True, alpha=0.4)
for ax in g.axes.flat:
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks([])
g.add_legend(title="Type d'École")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Distribution des Scores en Lecture par Pays et Type d’École", fontsize=16, fontweight="bold")
resize_and_save(g.fig, "DistributionPays_Lecture_SchoolType.png")

# ==== DISTRIBUTION PAR PAYS AVEC MOYENNES ====

# Moyennes par pays et type d’école – Mathématiques
math_means = data.groupby(["PAYS", "school_type"])["MATHS_PV5"].mean().reset_index()

# FacetGrid Mathématiques avec moyennes
g = sns.FacetGrid(data, col="PAYS", hue="school_type", col_wrap=4, height=4, sharey=False, palette=school_palette)
g.map(sns.kdeplot, "MATHS_PV5", common_norm=False, fill=True, alpha=0.4)

# Ajout des lignes de moyennes
for ax, country in zip(g.axes.flat, g.col_names):
    for school_type in data["school_type"].unique():
        mean_val = math_means[(math_means["PAYS"] == country) & (math_means["school_type"] == school_type)]["MATHS_PV5"]
        if not mean_val.empty:
            ax.axvline(mean_val.values[0], linestyle="--", color=school_palette[school_type], linewidth=1)

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks([])

g.add_legend(title="Type d'École")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Distribution des Scores en Mathématiques par Pays et Type d’École (avec Moyennes)", fontsize=16, fontweight="bold")
resize_and_save(g.fig, "DistributionPays_Maths_SchoolType_Moyennes.png")


# Moyennes par pays et type d’école – Lecture
read_means = data.groupby(["PAYS", "school_type"])["LECT_PV5"].mean().reset_index()

# FacetGrid Lecture avec moyennes
g = sns.FacetGrid(data, col="PAYS", hue="school_type", col_wrap=4, height=4, sharey=False, palette=school_palette)
g.map(sns.kdeplot, "LECT_PV5", common_norm=False, fill=True, alpha=0.4)

# Ajout des lignes de moyennes
for ax, country in zip(g.axes.flat, g.col_names):
    for school_type in data["school_type"].unique():
        mean_val = read_means[(read_means["PAYS"] == country) & (read_means["school_type"] == school_type)]["LECT_PV5"]
        if not mean_val.empty:
            ax.axvline(mean_val.values[0], linestyle="--", color=school_palette[school_type], linewidth=1)

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks([])

g.add_legend(title="Type d'École")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Distribution des Scores en Lecture par Pays et Type d’École (avec Moyennes)", fontsize=16, fontweight="bold")
resize_and_save(g.fig, "DistributionPays_Lecture_SchoolType_Moyennes.png")
