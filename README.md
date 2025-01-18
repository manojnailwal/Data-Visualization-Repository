# **Data Visualization Repository**

Welcome to the **Data Visualization** repository! In this repository, you'll find various examples of creating different types of data visualizations using popular Python libraries: **Matplotlib**, **Plotly**, **Bokeh**, and **Seaborn**.

## **What is in this repository?**

This repository contains a collection of charts and plots created using different visualization techniques for representing datasets. The plots cover a wide range of data visualization techniques to help visualize patterns, distributions, and relationships within datasets.

### **Key Visualizations in the Repository**:
1. **Bar Charts**: 
   - Represent the counts of different categories.
   - Used in **Matplotlib**, **Plotly**, **Seaborn**, and **Bokeh** to display discrete data.

2. **Scatter Plots**: 
   - Used to represent relationships between two numerical variables.
   - Visualized with **Plotly**, **Matplotlib**, and **Bokeh**.

3. **Histograms**:
   - Display the distribution of numerical data.
   - Visualized using **Bokeh** and **Matplotlib**.

4. **Line Plots**:
   - Show the trends or changes in a variable over time or other continuous data.
   - Implemented with **Plotly** and **Matplotlib**.

5. **Heatmaps**:
   - Used to show the intensity of values across two dimensions.
   - Created using **Bokeh** and **Seaborn**.

6. **3D Plots**:
   - Visualize data in three dimensions.
   - Implemented using **Plotly** and **Matplotlib**.

7. **Violin Plots**:
   - Display the distribution of a numerical variable across different categories.
   - Visualized using **Seaborn**.

8. **Bubble Charts**:
   - Visualize three dimensions of data: X, Y, and the size of the bubble.
   - Created using **Plotly**.

### **What did I do in this repository?**

- **Data Visualization with Python**: I created a wide variety of interactive and static visualizations to represent data patterns and distributions. This includes using different libraries that support both interactive and static plotting.
- **Learning and Experimenting**: I experimented with different types of plots like bar charts, line plots, scatter plots, and heatmaps to visualize the data from various perspectives.
- **Library Exploration**: I explored the functionality and unique features of each library:
    - **Matplotlib**: For basic static plots and customizations.
    - **Plotly**: For interactive, web-ready visualizations.
    - **Bokeh**: For interactive visualizations in Python, especially suited for web apps.
    - **Seaborn**: For statistical plotting and advanced visualization techniques built on top of Matplotlib.
- **Dataset Creation and Handling**: I generated random and synthetic datasets, as well as worked with sample data, to create meaningful visualizations that demonstrate the utility of each plot type.

---

## **Libraries Used**:

- **Matplotlib**: A comprehensive library for creating static, animated, and interactive plots in Python.
- **Plotly**: A graphing library for creating interactive plots that are easy to share and embed.
- **Bokeh**: A powerful library for creating interactive, web-ready visualizations.
- **Seaborn**: A statistical data visualization library based on Matplotlib.

---

## **How to Use This Repository**:

### 1. **Clone the Repository**:
```bash
git clone https://github.com/yourusername/data-visualization-repository.git
cd data-visualization-repository
```

### 2. **Install Required Libraries**:
Ensure that you have the necessary libraries installed. You can install the required dependencies using `pip`:
```bash
pip install matplotlib plotly seaborn bokeh pandas numpy
```

### 3. **Run the Visualizations**:
- Open the Python scripts or Jupyter notebooks (if available) in your favorite Python editor.
- Each script contains a visualization, so simply run them to generate the corresponding plots.

### 4. **Interactive Plots**:
For **Plotly** and **Bokeh** plots, ensure that you're working in an environment (such as Jupyter or a local environment) that supports interactive visualization. You can use `output_notebook()` for inline visualization in Jupyter notebooks.

---

## **Example Visualizations**:

### 1. **Matplotlib Example**:
```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a simple line plot
plt.plot(x, y, label="Sine Wave")
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Sine Wave')
plt.legend()
plt.show()
```

### 2. **Plotly Example**:
```python
import plotly.express as px

# Sample data
data = {'X': [1, 2, 3, 4, 5], 'Y': [10, 11, 12, 13, 14]}
df = pd.DataFrame(data)

# Create a scatter plot
fig = px.scatter(df, x='X', y='Y', title='Simple Scatter Plot')
fig.show()
```

### 3. **Bokeh Example**:
```python
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

output_notebook()

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 11, 12, 13, 14]

# Create a Bokeh scatter plot
p = figure(title="Simple Scatter Plot", x_axis_label='X', y_axis_label='Y')
p.scatter(x, y, size=10, color="blue", legend_label="Data Points")

show(p)
```

### 4. **Seaborn Example**:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset
tips = sns.load_dataset('tips')

# Create a violin plot
sns.violinplot(x='day', y='total_bill', data=tips)
plt.title('Distribution of Total Bill by Day')
plt.show()
```

---

## **Contributions**:
If you'd like to contribute to this repository, feel free to open a **pull request** with your suggested changes, additional visualizations, or improvements.

---

## **License**:
This repository is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for more details.
