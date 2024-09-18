import numpy as np # type: ignore
from scipy.optimize import minimize # type: ignore
import pandas as pd # type: ignore
import streamlit as st # type: ignore
import matplotlib.pyplot as plt # type: ignore
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # type: ignore
import sympy as sp # type: ignore
import statsmodels.api as sm # type: ignore
from statsmodels.regression import linear_model # type: ignore

class Bottle:
    def __init__(self, volume, max_height=None, max_side_length=None):
        self.volume = volume
        self.max_height = max_height
        self.max_side_length = max_side_length
    
    def optimise_result(self):
        raise NotImplementedError("This method should be overridden by subclasses")

class TrianglePrism(Bottle):
    def __init__(self, volume, max_height=None, max_side_length=None):
        super().__init__(volume, max_height, max_side_length)
    
    def adjust_parameters(self, s_opt, h_pr_opt, h_p_opt):
        total_height = h_p_opt + h_pr_opt
        if self.max_height is not None and total_height > self.max_height:
            h_p_opt = self.max_height - h_pr_opt
            a = h_pr_opt + h_p_opt / 3
            b = h_p_opt / 3
            c = h_p_opt / 3 - (self.volume * 4 / np.sqrt(3))
            det = np.sqrt(b**2 - 4 * a * c)
            s1 = -b + det / 2 * a
            s2 = -b - det / 2 * a
            if s1 > 0:
                s_opt = s1
            else:
                s_opt = s2
        return s_opt, h_pr_opt, h_p_opt
    
    def optimise_result(self):
        def surface_area(params):
            s, h_p = params
            h_pr = (self.volume - (1 / 3) * np.sqrt(3) / 4 * h_p * (s**2 + s * 2 + 4)) / (np.sqrt(3) / 4 * s**2)
            if h_pr < 0 or s <= 1:
                return np.inf
            A_base = np.sqrt(3) * s**2 / 4
            A_side_pr = 3 * s * h_pr
            A_side_p = 3 * s * np.sqrt(h_p**2 + (s / 2 - 1)**2)
            A = A_base + A_side_pr + A_side_p
            return A

        bounds = [(1.01, self.max_side_length if self.max_side_length else 100), (0, 100)]
        initial_guess = [2, 10]
        result = minimize(surface_area, initial_guess, bounds=bounds)
        s_opt, h_p_opt = result.x
        h_pr_opt = (self.volume - (1 / 3) * np.sqrt(3) / 4 * h_p_opt * (s_opt**2 + s_opt * 2 + 4)) / (np.sqrt(3) / 4 * s_opt**2)
        s_opt, h_pr_opt, h_p_opt = self.adjust_parameters(s_opt, h_pr_opt, h_p_opt)
        A = np.sqrt(3) / 4 * s_opt**2 + 3 * s_opt * np.sqrt(h_p_opt**2 + (s_opt / 2 - 1)**2) + 3 * s_opt * h_pr_opt
        return s_opt, h_pr_opt, h_p_opt, A

class SquarePrism(Bottle):
    def __init__(self, volume, max_height=None, max_side_length=None):
        super().__init__(volume, max_height, max_side_length)
    
    def adjust_parameters(self, s_opt, h_pr_opt, h_p_opt):
        total_height = h_p_opt + h_pr_opt
        if self.max_height is not None and total_height > self.max_height:
            h_p_opt = self.max_height - h_pr_opt
            a = h_pr_opt + h_p_opt / 3
            b = h_p_opt / 3
            c = h_p_opt / 3 - self.volume
            det = np.sqrt(b**2 - 4 * a * c)
            s1 = -b + det / 2 * a
            s2 = -b - det / 2 * a
            if s1 > 0:
                s_opt = s1
            else:
                s_opt = s2
        return s_opt, h_pr_opt, h_p_opt
    
    def optimise_result(self):
        def surface_area(parameters):
            s, h_p = parameters
            h_pr = (self.volume - (h_p * (s**2 + s + 1) / 3)) / s**2
            surface_area = s**2 + 4 * s * h_pr + 2 * (s + 1) * (np.sqrt((s - 1)**2 + h_p**2))
            return surface_area

        initial_guess = [2, 15]
        result = minimize(surface_area, initial_guess, bounds=[(1, self.max_side_length), (1, 100)])
        s_opt, h_p_opt = result.x
        h_pr_opt = (self.volume - (h_p_opt * (s_opt**2 + s_opt + 1) / 3)) / s_opt**2
        s_opt, h_pr_opt, h_p_opt = self.adjust_parameters(s_opt, h_pr_opt, h_p_opt)
        A = s_opt**2 + 4 * s_opt * h_pr_opt + 2 * (s_opt + 1) * (np.sqrt((s_opt - 1)**2 + h_p_opt**2))
        return s_opt, h_pr_opt, h_p_opt, A

class PentagonPrism(Bottle):
    def __init__(self, volume, max_height=None, max_side_length=None):
        super().__init__(volume, max_height, max_side_length)
    
    def adjust_parameters(self, s_opt, h_pr_opt, h_p_opt):
        total_height = h_p_opt + h_pr_opt
        const = np.sqrt(5 * (5 + 2 * np.sqrt(5))) / 4
        if self.max_height is not None and total_height > self.max_height:
            h_p_opt = self.max_height - h_pr_opt
            a = h_pr_opt + h_p_opt / 3
            b = h_p_opt / 3
            c = h_p_opt / 3 - self.volume / const
            det = np.sqrt(b**2 - 4 * a * c)
            s1 = -b + det / 2 * a
            s2 = -b - det / 2 * a
            if s1 > 0:
                s_opt = s1
            else:
                s_opt = s2
        return s_opt, h_pr_opt, h_p_opt
    
    def optimise_result(self):
        const = np.sqrt(5 * (5 + 2 * np.sqrt(5))) / 4
        def surface_area(parameters):
            s, h_p = parameters
            h_pr = (self.volume - (const / 3) * (s**2 + s + 1)) / (const * s**2)
            surface_area = const * s**2 + 5 * s * h_pr + 2.5 * (s + 1) * (np.sqrt((s - 1)**2 + h_p**2))
            return surface_area

        initial_guess = [2, 15]
        result = minimize(surface_area, initial_guess, bounds=[(1, self.max_side_length), (1, 100)])
        s_opt, h_p_opt = result.x
        h_pr_opt = (self.volume - (const / 3) * (s_opt**2 + s_opt + 1)) / (const * s_opt**2)
        s_opt, h_pr_opt, h_p_opt = self.adjust_parameters(s_opt, h_pr_opt, h_p_opt)
        A = (const * s_opt**2) + (5 * s_opt * h_pr_opt) + (2.5 * (s_opt + 1) * (np.sqrt((s_opt - 1)**2 + h_p_opt**2)))
        return s_opt, h_pr_opt, h_p_opt, A

class HexagonPrism(Bottle):
    def __init__(self, volume, max_height=None, max_side_length=None):
        super().__init__(volume, max_height, max_side_length)
    
    def adjust_parameters(self, s_opt, h_pr_opt, h_p_opt):
        total_height = h_p_opt + h_pr_opt
        const = 3 * np.sqrt(3) / 2
        if self.max_height is not None and total_height > self.max_height:
            h_p_opt = self.max_height - h_pr_opt
            a = h_pr_opt + h_p_opt / 3
            b = h_p_opt / 3
            c = h_p_opt / 3 - self.volume / const
            det = np.sqrt(b**2 - 4 * a * c)
            s1 = -b + det / 2 * a
            s2 = -b - det / 2 * a
            if s1 > 0:
                s_opt = s1
            else:
                s_opt = s2
        return s_opt, h_pr_opt, h_p_opt
    
    def optimise_result(self):
        const = 3 * np.sqrt(3) / 2
        def surface_area(parameters):
            s, h_p = parameters
            h_pr = (self.volume - (const / 3) * (s**2 + s + 1)) / (const * s**2)
            surface_area = const * s**2 + 6 * s * h_pr + 3 * (s + 1) * (np.sqrt((s - 1)**2 + h_p**2))
            return surface_area

        initial_guess = [2, 15]
        result = minimize(surface_area, initial_guess, bounds=[(1, self.max_side_length), (1, 100)])
        s_opt, h_p_opt = result.x
        h_pr_opt = (self.volume - (const / 3) * (s_opt**2 + s_opt + 1)) / (const * s_opt**2)
        s_opt, h_pr_opt, h_p_opt = self.adjust_parameters(s_opt, h_pr_opt, h_p_opt)
        A = (const * s_opt**2) + (6 * s_opt * h_pr_opt) + (3 * (s_opt + 1) * (np.sqrt((s_opt - 1)**2 + h_p_opt**2)))
        return s_opt, h_pr_opt, h_p_opt, A

class CylinderCone(Bottle):
    def __init__(self, volume, max_height=None, max_side_length=None):
        super().__init__(volume, max_height, max_side_length)
        
    def adjust_parameters (self, r_opt, h_opt, h_cone_opt):
        total_height = h_cone_opt + h_opt
        
        if self.max_height is not None and total_height > self.max_height:
            h_cone_opt = self.max_height - h_opt
            
            a = h_opt + h_cone_opt/3
            b = h_cone_opt/3
            c = h_cone_opt/3 - self.volume/np.pi
            det = np.sqrt(b**2 - 4*a*c)
            
            r1 = - b + det / 2*a
            r2 = - b - det / 2*a
            
            if r1 > 0: 
                r_opt = r1
            else: 
                r_opt = r2
        return r_opt, h_opt, h_cone_opt
    
    def optimise_result(self):
        def surface_area (parameters):
            r, h_cone = parameters
            h = (self.volume - (1/3) * np.pi * h_cone * (r**2 + r + 1)) / (np.pi * r**2)
            surface_area = np.pi*(r+1)*np.sqrt((r-1)**2+h_cone**2)+np.pi*(r**2+1) + 2*np.pi*r*h
            return surface_area

        initial_guess = [2,15]
        result = minimize(surface_area, initial_guess, bounds = [(1, self.max_side_length),(1,100)])
        r_opt, h_cone_opt = result.x
        h_opt = (self.volume - (1/3) * np.pi * h_cone_opt * (r_opt**2 + r_opt + 1)) / (np.pi * r_opt**2)
        r_opt, h_opt, h_cone_opt = self.adjust_parameters(r_opt, h_opt, h_cone_opt)
        
        A = np.pi*(r_opt+1)*np.sqrt((r_opt-1)**2+h_cone_opt**2) + np.pi*(r_opt**2+1) + 2*np.pi*r_opt*h_opt
        
        return r_opt, h_opt, h_cone_opt, A

def create_dataframe(volume, max_height, max_side_length):
    bottles = [
        TrianglePrism(volume, max_height, max_side_length),
        SquarePrism(volume, max_height, max_side_length),
        PentagonPrism(volume, max_height, max_side_length),
        HexagonPrism(volume, max_height, max_side_length),
        CylinderCone(volume, max_height, max_side_length)
    ]

    data = []

    for bottle in bottles:
        s_opt, h_pr_opt, h_p_opt, min_surface_area = bottle.optimise_result()
        data.append([bottle.__class__.__name__, s_opt, h_pr_opt, h_p_opt, min_surface_area])

    df = pd.DataFrame(data, columns=["Shape", "Optimized Side Length/Radius", "Height of Bottom Section", "Height of Top Section", "Minimum Surface Area"])
    return df

st.title("Bottle Shape Optimization")
st.header("Optimization Input and Results")

st.subheader("Input Parameters")
volume = st.number_input("Volume", value=1000)
max_height = st.number_input("Max Height", value=30)
max_side_length = st.number_input("Max Side Length", value=20)

df = create_dataframe(volume, max_height, max_side_length)
st.write("Optimization Results")
st.dataframe(df)

st.write("Optimized Surface Area vs. Number of Vertices")
vertices = [3, 4, 5, 6, np.inf]
plt.figure(figsize=(10, 6))
plt.plot(vertices, df['Minimum Surface Area'])
plt.xlabel('Number of Vertices')
plt.ylabel('Optimized Surface Area')
plt.title('Optimized Surface Area vs. Number of Vertices')
plt.grid(True)
st.pyplot(plt)

st.write("3D Shape Visualization")
shape_classes = [TrianglePrism, SquarePrism, PentagonPrism, HexagonPrism, CylinderCone]
shape_names = ['Triangle Prism', 'Square Prism', 'Pentagon Prism', 'Hexagon Prism', 'Cylinder-Cone']

def plot_shape(index, ax):
    ax.clear()
    shape_name = shape_names[index]
    dimensions = df.iloc[index, 1:4].values
    s = dimensions[0]
    h_pr = dimensions[1]
    h_p = dimensions[2]

    if shape_name == 'Triangle Prism':
            s, h_pr, h_p = dimensions
            s = 0.1 * s
            h_p = 0.1 * h_p
            h_pr = 0.1 * h_pr
            
            v_prism = np.array([[0, 0, 0], [s, 0, 0], [s/2, np.sqrt(3)*s/2, 0], [0, 0, h_pr], [s, 0, h_pr], [s/2, np.sqrt(3)*s/2, h_pr]])
            v_pyramid = np.array([[0, 0, h_pr], [s, 0, h_pr], [s/2, np.sqrt(3)*s/2, h_pr], [s/2, np.sqrt(3)/6, h_pr + h_p]])

            verts_prism = [[v_prism[0], v_prism[1], v_prism[2]], [v_prism[0], v_prism[1], v_prism[4], v_prism[3]], 
                        [v_prism[1], v_prism[2], v_prism[5], v_prism[4]], [v_prism[2], v_prism[0], v_prism[3], v_prism[5]]]
            
            verts_pyramid = [[v_pyramid[0], v_pyramid[1], v_pyramid[3]], [v_pyramid[1], v_pyramid[2], v_pyramid[3]],
                            [v_pyramid[2], v_pyramid[0], v_pyramid[3]], [v_pyramid[0], v_pyramid[1], v_pyramid[2]]]

            ax.add_collection3d(Poly3DCollection(verts_prism, facecolors='b', linewidths=1, edgecolors='r', alpha=.3))
            ax.add_collection3d(Poly3DCollection(verts_pyramid, facecolors='r', linewidths=1, edgecolors='r', alpha=.3))

    elif shape_name == 'Square Prism':
            s, h_pr, h_p = dimensions
            s = 0.1 * s
            h_p = 0.1 * h_p
            h_pr = 0.1 * h_pr
            
            v_prism = np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0], [0, 0, h_pr], [s, 0, h_pr], [s, s, h_pr], [0, s, h_pr]])
            v_pyramid = np.array([[0, 0, h_pr], [s, 0, h_pr], [s, s, h_pr], [0, s, h_pr], [s/2, s/2, h_pr + h_p]])

            verts_prism = [[v_prism[0], v_prism[1], v_prism[5], v_prism[4]], [v_prism[1], v_prism[2], v_prism[6], v_prism[5]], 
                        [v_prism[2], v_prism[3], v_prism[7], v_prism[6]], [v_prism[3], v_prism[0], v_prism[4], v_prism[7]],
                        [v_prism[0], v_prism[1], v_prism[2], v_prism[3]], [v_prism[4], v_prism[5], v_prism[6], v_prism[7]]]
            
            verts_pyramid = [[v_pyramid[0], v_pyramid[1], v_pyramid[4]], [v_pyramid[1], v_pyramid[2], v_pyramid[4]], 
                            [v_pyramid[2], v_pyramid[3], v_pyramid[4]], [v_pyramid[3], v_pyramid[0], v_pyramid[4]]]

            ax.add_collection3d(Poly3DCollection(verts_prism, facecolors='b', linewidths=1, edgecolors='r', alpha=.3))
            ax.add_collection3d(Poly3DCollection(verts_pyramid, facecolors='r', linewidths=1, edgecolors='r', alpha=.3))

    elif shape_name == 'Pentagon Prism':
            s, h_pr, h_p = dimensions
            s = 0.1 * s
            h_p = 0.1 * h_p
            h_pr = 0.1 * h_pr
            angle = np.linspace(0, 2 * np.pi, 6)[:-1]
            v_prism = np.array([[np.cos(a)*s, np.sin(a)*s, 0] for a in angle] + [[np.cos(a)*s, np.sin(a)*s, h_pr] for a in angle])
            v_pyramid = np.array([[np.cos(a)*s, np.sin(a)*s, h_pr] for a in angle] + [[0, 0, h_pr + h_p]])

            verts_prism = [[v_prism[i], v_prism[(i+1)%5], v_prism[(i+1)%5+5], v_prism[i+5]] for i in range(5)]
            verts_prism.append([v_prism[i] for i in range(5)])
            verts_prism.append([v_prism[i+5] for i in range(5)])
            
            verts_pyramid = [[v_pyramid[i], v_pyramid[(i+1)%5], v_pyramid[5]] for i in range(5)]

            ax.add_collection3d(Poly3DCollection(verts_prism, facecolors='b', linewidths=1, edgecolors='r', alpha=.3))
            ax.add_collection3d(Poly3DCollection(verts_pyramid, facecolors='r', linewidths=1, edgecolors='r', alpha=.3))

    elif shape_name == 'Hexagon Prism':
            s, h_pr, h_p = dimensions
            s = 0.1 * s
            h_p = 0.1 * h_p
            h_pr = 0.1 * h_pr
            angle = np.linspace(0, 2 * np.pi, 7)[:-1]
            v_prism = np.array([[np.cos(a)*s, np.sin(a)*s, 0] for a in angle] + [[np.cos(a)*s, np.sin(a)*s, h_pr] for a in angle])
            v_pyramid = np.array([[np.cos(a)*s, np.sin(a)*s, h_pr] for a in angle] + [[0, 0, h_pr + h_p]])

            verts_prism = [[v_prism[i], v_prism[(i+1)%6], v_prism[(i+1)%6+6], v_prism[i+6]] for i in range(6)]
            verts_prism.append([v_prism[i] for i in range(6)])
            verts_prism.append([v_prism[i+6] for i in range(6)])
            
            verts_pyramid = [[v_pyramid[i], v_pyramid[(i+1)%6], v_pyramid[6]] for i in range(6)]

            ax.add_collection3d(Poly3DCollection(verts_prism, facecolors='b', linewidths=1, edgecolors='r', alpha=.3))
            ax.add_collection3d(Poly3DCollection(verts_pyramid, facecolors='r', linewidths=1, edgecolors='r', alpha=.3))
    
    elif shape_name == 'Cylinder-Cone':
            r, h_cy, h_c = dimensions
            z_cy = np.linspace(0, h_cy, 100)
            z_c = np.linspace(h_cy, h_cy + h_c, 100)
            x_cy = r * np.cos(np.linspace(0, 2*np.pi, 100))
            y_cy = r * np.sin(np.linspace(0, 2*np.pi, 100))
            x_c = np.linspace(r, 1, 100)[:, np.newaxis] * np.cos(np.linspace(0, 2*np.pi, 100))
            y_c = np.linspace(r, 1, 100)[:, np.newaxis] * np.sin(np.linspace(0, 2*np.pi, 100))
            
            ax.plot_surface(x_cy, y_cy, z_cy[:, np.newaxis], color='b', alpha=0.6)
            ax.plot_surface(x_c, y_c, z_c[:, np.newaxis], color='r', alpha=0.6)

    ax.set_title(f"{shape_name} Bottle")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_dimension = max(s, h_pr + h_p)
    ax.set_xlim(-max_dimension, max_dimension)
    ax.set_ylim(-max_dimension, max_dimension)
    ax.set_zlim(0, max_dimension)
    ax.set_box_aspect([1,1,1])

current_shape_index = st.selectbox("Select Shape", range(len(shape_names)), format_func=lambda x: shape_names[x])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_shape(current_shape_index, ax)
st.pyplot(fig)
