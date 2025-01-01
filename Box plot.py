# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Define modern colors for publication
modern_palette = {
    "Male": "#4B8BBE",  # Modern blue
    "Female": "#FF8C42"  # Soft orange
}

# Set file path for data
data_path = "C:/Users/massa/Desktop/MiniProjet/projet passec/Data/PASEC2019_GRADE6_TREAT.dta"

# Load the data (resolve encoding issue with latin-1 fallback)
data = pd.read_stata(data_path, convert_categoricals=False)

# Replace numeric values in 'sexe' with descriptive labels
data['sexe'] = data['sexe'].map({1.0: 'Male', 2.0: 'Female'})

# Remove rows with missing 'sexe' values
data = data.dropna(subset=['sexe'])

# Ensure 'PAYS' and 'sexe' are categorical
data['PAYS'] = data['PAYS'].astype('category')
data['sexe'] = data['sexe'].astype('category')

# Smaller font sizes for axes
small_font = {'fontsize': 10}

# Calculate gender counts and percentages
gender_counts = data['sexe'].value_counts(normalize=True) * 100
male_percentage = round(gender_counts.get("Male", 0), 1)
female_percentage = round(gender_counts.get("Female", 0), 1)

# Custom legend labels
legend_labels = [f"Male ({male_percentage}%)", f"Female ({female_percentage}%)"]

# Define save path for results
save_path = "C:/Users/massa/Desktop/MiniProjet/projet passec/Figure python/"
os.makedirs(save_path, exist_ok=True)

# Plot 1: Math PV5 Distribution by Country and Gender
plt.figure(figsize=(8, 5))  # Échelle réduite
sns.boxplot(
    data=data,
    x="PAYS",
    y="MATHS_PV5",
    hue="sexe",
    palette=modern_palette
)
plt.title("Distribution des Scores en Mathématiques", fontweight="bold", fontsize=12)
plt.xlabel("", **small_font)
plt.ylabel("Scores en Mathématiques", **small_font)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.legend(title="Genre", labels=legend_labels, fontsize=8, title_fontsize=10)

# Ajout de la signature et informations
plt.figtext(0.95, 0.01, "Auteur: MNS Awahid", fontsize=7, ha="right")
plt.figtext(0.95, 0.03, "Logiciel: Python", fontsize=7, ha="right")

plt.tight_layout()
plt.savefig(save_path + "Math_PV5_Signature.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 2: LECT PV5 Distribution by Country and Gender
plt.figure(figsize=(8, 5))  # Échelle réduite
sns.boxplot(
    data=data,
    x="PAYS",
    y="LECT_PV5",
    hue="sexe",
    palette=modern_palette
)
plt.title("Distribution des Scores en Lecture"
          , fontweight="bold", fontsize=12)
plt.xlabel("", **small_font)
plt.ylabel("Scores en Lecture", **small_font)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.legend(title="Genre", labels=legend_labels, fontsize=8, title_fontsize=10)

# Ajout de la signature et informations
plt.figtext(0.95, 0.01, "Auteur: MNS Awahid", fontsize=7, ha="right")
plt.figtext(0.95, 0.03, "Logiciel: Python", fontsize=7, ha="right")

plt.tight_layout()
plt.savefig(save_path + "Lecture_PV5_Signature.png", dpi=300, bbox_inches="tight")
plt.show()

# Shared legend setup (combining both plots without individual legends)
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Graphique 1 : Mathématiques PV5
sns.boxplot(
    data=data,
    x="PAYS",
    y="MATHS_PV5",
    hue="sexe",
    ax=axes[0],
    palette=modern_palette
)
axes[0].set_title("Mathématiques", fontweight="bold", fontsize=10)
axes[0].set_xlabel("", fontsize=8)
axes[0].set_ylabel("Scores", fontsize=8)
axes[0].tick_params(axis='x', rotation=45, labelsize=8)
axes[0].tick_params(axis='y', labelsize=8)
axes[0].get_legend().remove()

# Graphique 2 : Lecture PV5
sns.boxplot(
    data=data,
    x="PAYS",
    y="LECT_PV5",
    hue="sexe",
    ax=axes[1],
    palette=modern_palette
)
axes[1].set_title("Lecture", fontweight="bold", fontsize=10)
axes[1].set_xlabel("", fontsize=8)
axes[1].set_ylabel("", fontsize=8)
axes[1].tick_params(axis='x', rotation=45, labelsize=8)
axes[1].tick_params(axis='y', labelsize=8)
axes[1].get_legend().remove()

# Ajout de la légende partagée sous les graphiques
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, legend_labels, title="Genre", loc="lower center", ncol=2, fontsize=8, title_fontsize=8)

# Ajustement des marges pour la légende
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.subplots_adjust(hspace=0.1, wspace=0.1)

# Ajout des informations globales
plt.figtext(0.95, 0.01, "Auteur: MNS Awahid ", fontsize=7, ha="right")
plt.figtext(0.95, 0.03, "Logiciel: Python", fontsize=7, ha="right")

# Sauvegarde du graphique combiné
plt.savefig(
    save_path + "Math_Lecture_PV5_Combined_Signature.png",
    dpi=300, bbox_inches="tight", transparent=False, format="png"
)

plt.show()
