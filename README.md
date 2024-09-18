# bottle-optimization

# Bottle Shape Optimization - README

## Web Link:
https://alvinyadav-bottle-optimization-home-uqzmvh.streamlit.app/

## Overview
This project implements an optimization tool to determine the most efficient bottle shape with a given volume, subject to constraints on maximum height and side length. The program computes and visualizes various bottle shapes to minimize surface area, which can be useful in packaging and material conservation applications. The shapes included are:

- Triangle Prism
- Square Prism
- Pentagon Prism
- Hexagon Prism
- Cylinder-Cone

The project is built using Python, leveraging optimization libraries and visualization tools to provide both numerical results and 3D shape visualizations.

## Features
- **Surface Area Optimization**: The tool minimizes the surface area for each bottle shape given a specific volume, ensuring that material usage is minimized.
- **Adjustable Constraints**: You can set maximum values for height and side length.
- **Interactive Visualization**: The optimized shapes can be viewed through 3D visualizations, and their surface area performance is plotted against the number of vertices.
- **Streamlit Integration**: A user-friendly web interface is provided using Streamlit, allowing easy input of parameters and viewing of results.

## Requirements
The program uses the following Python libraries:
- `numpy`
- `scipy`
- `pandas`
- `streamlit`
- `matplotlib`
- `sympy`
- `statsmodels`

Ensure all dependencies are installed before running the project:
```bash
pip install numpy scipy pandas streamlit matplotlib sympy statsmodels
```

## Usage
To run the optimization tool, use the following steps:

1. Clone the repository and navigate to the project directory.
2. Run the Streamlit app:
   ```bash
   streamlit run Home.py
   ```
3. Use the web interface to input the desired **volume**, **maximum height**, and **maximum side length**.
4. View the optimization results in tabular form, which includes:
   - Optimized side length (or radius for cylinder-cone)
   - Height of the bottom and top sections
   - Minimum surface area for each shape

5. View the graph that plots the optimized surface area against the number of vertices of each shape.
6. Visualize the 3D structure of each shape by selecting from the available options.

## Program Structure

### Classes
- **Bottle**: The base class for all bottle shapes.
- **TrianglePrism, SquarePrism, PentagonPrism, HexagonPrism, CylinderCone**: Subclasses that implement the optimization logic for each respective shape.

Each subclass contains:
- **adjust_parameters**: Adjusts side length and height to ensure constraints are met.
- **optimise_result**: Performs optimization to minimize surface area.

### Streamlit Interface
The app provides a user-friendly interface to input parameters, run optimizations, and visualize results in both tabular and graphical formats. The app is divided into sections for input parameters, optimization results, and visualizations.

### Visualization
- A line plot shows the optimized surface area versus the number of vertices for each shape.
- 3D shape visualizations are generated using `matplotlib` and displayed interactively.

## Example
To compute the optimal bottle shape for a volume of 1000 units with a maximum height of 30 units and a maximum side length of 20 units:
- Enter these values in the Streamlit app.
- The results will show the optimal parameters for each shape, including the side length, height, and minimized surface area.
- The 3D visualization and surface area plot will update accordingly.
