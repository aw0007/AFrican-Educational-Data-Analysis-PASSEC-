# PASEC Data Visualization: Exploring Educational Test Scores by Gender and Country

This repository contains two Python scripts for analyzing and visualizing educational test score data from the **PASEC 2019** dataset. The analysis focuses on gender and country-level variations in mathematics and reading test scores for Grade 6 students.

---

## Repository Structure

- **Data Folder**:
  - Contains the input dataset: `PASEC2019_GRADE6_TREAT.dta`.
- **Scripts Folder**:
  - `Box plot.py`: Generates box plots to compare test scores by gender and country.
  - `Distribution.py`: Produces KDE (Kernel Density Estimate) plots and faceted visualizations for detailed score distributions.
- **Results Folder**:
  - Contains the output figures and visualizations generated by the scripts.

---

## Tools and Libraries Used

- **Python**:
  - `pandas`: For data preprocessing and analysis.
  - `matplotlib`: For creating plots and visualizations.
  - `seaborn`: For advanced data visualization.
  - `os`: For managing file directories.
  - `Pillow (PIL)`: For resizing and optimizing output images.

---

## Key Features and Outputs

### 1. **Box plot.py**

This script generates box plots to visualize the distribution of test scores by country and gender for both mathematics and reading.

#### Key Outputs:
- **Box Plots**:
  - `BoxPlot_MATHS_PV5.png`: Mathematics scores by country and gender.
  - `BoxPlot_LECT_PV5.png`: Reading scores by country and gender.
- **Combined Box Plot**:
  - `Math_Lecture_PV5_Combined_Signature.png`: Combined box plots for mathematics and reading scores with a shared legend.

#### Key Features:
- Custom legends displaying percentages of male and female students.
- Rotated x-axis labels for better readability of country names.
- Inclusion of author signature and software details in the visualizations.

---

### 2. **Distribution.py**

This script focuses on visualizing score distributions using KDE plots and faceted grids, providing insights into overall and country-specific trends by gender.

#### Key Outputs:
- **Overall KDE Plots**:
  - `Overall_Distribution_MATHS_PV5.png`: Overall distribution of mathematics scores by gender.
  - `Overall_Distribution_LECT_PV5.png`: Overall distribution of reading scores by gender.
- **Faceted KDE Plots**:
  - `Facet_Distribution_MATHS_PV5.png`: Mathematics score distributions for each country by gender.
  - `Facet_Distribution_LECT_PV5.png`: Reading score distributions for each country by gender.

#### Key Features:
- Faceted visualizations for country-specific score distributions.
- Shared legends for improved readability.
- High-resolution outputs optimized for publications.

---

## How to Run the Scripts

1. **Set Up Input Files**:
   - Place the dataset `PASEC2019_GRADE6_TREAT.dta` in the specified data directory.

2. **Run Each Script**:
   - Execute `Box plot.py` for box plots and combined visualizations.
   - Execute `Distribution.py` for KDE plots and faceted grids.

3. **View and Save Outputs**:
   - Outputs will be saved in the `Figure python` directory within the project folder.

---

## Sample Outputs

### Box Plot of Mathematics Scores by Country and Gender (from `Box plot.py`)
![Box Plot Math](Figure python/BoxPlot_MATHS_PV5.png)

### Overall KDE Plot for Mathematics Scores by Gender (from `Distribution.py`)
![KDE Plot Math](Figure python/Overall_Distribution_MATHS_PV5.png)

---

## Author and Credits

- **Author**: MNS Awahid  
- **Dataset**: [PASEC 2019](https://www.pasec.confemen.org)

