import streamlit as st # type: ignore

st.header("Home")
st.write("""
    Welcome to the Bottle Shape Optimization App!
    
    This application helps you find the optimal shape for a bottle given a specific volume, maximum height, and maximum side length. 
    You can choose from different bottle shapes including triangular, square, pentagonal, hexagonal prisms, and a cylinder-cone combination.
    
    Navigate to the 'Optimization' page to input your parameters and get the results.
    If you need assistance, check the 'Help' page for more information.
    """)
