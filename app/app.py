import streamlit as st

st.title("My Hackathon Project")
st.write("AI + Language track")

text = st.text_area("Type something:")

if st.button("Go"):
    # The AI part goes here once we pick an idea.
    st.write("You typed:", text)
