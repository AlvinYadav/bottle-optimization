import streamlit as st  # type: ignore

# Title and introductory message
st.title("📦 Bottle Shape Optimization App")
st.write("""
    Welcome to the **Bottle Shape Optimization App**! 

    This app allows you to find the most efficient bottle shape for a given volume, height, and side length.
    Whether you're a designer, engineer, or just curious, our tool will help you explore various geometric shapes to optimize your design.
""")

# Visual breakdown of shapes
st.subheader("🔢 Available Shapes")
st.write("""
    You can optimize bottle shapes from the following options:
    
    - 🔺 Triangular Prism
    - ◼️ Square Prism
    - ⬟ Pentagonal Prism
    - ⬢ Hexagonal Prism
    - ⚪ Cylinder-Cone Combination
""")

# Instructions for next steps
st.subheader("🚀 Get Started")
st.write("""
    1. Head over to the **'Optimization'** page to enter the parameters for your bottle design (volume, height, side length).
    2. Get the optimized shape that maximizes volume efficiency for your constraints.
    3. View the results and visualize the shape.
""")

# Help and support section
st.subheader("❓ Need Help?")
st.write("""
    Visit the **'Help'** page for a guide on how to use the app and more details on each shape.
""")
