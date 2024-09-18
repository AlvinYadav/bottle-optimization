import streamlit as st  # type: ignore  

st.header("Help")
st.write("""
    ## How to Use the Bottle Shape Optimization App
    
    1. **Home Page**: Get an overview of what this application does.
    2. **Optimization Page**: Input your parameters such as volume, maximum height, and maximum side length. View the optimization results and visualize the shapes.
    3. **Help Page**: Find detailed instructions and explanations on how to use the app.
    
    ### Input Parameters
    - **Volume**: The volume of the bottle you want to design.
    - **Max Height**: The maximum allowable height of the bottle.
    - **Max Side Length**: The maximum allowable side length of the bottle's base shape.
    
    ### Results
    - The results table will show the optimized dimensions for each shape and the minimum surface area.
    - A plot showing the relationship between the number of vertices and the optimized surface area.
    - 3D visualizations of the optimized shapes.
    
    ### Shape Selection
    - Use the shape selection dropdown to switch between different bottle shapes and see their 3D visualizations.

    """)