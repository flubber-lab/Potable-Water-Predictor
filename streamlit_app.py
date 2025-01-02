import streamlit as st
import numpy as np
import pickle


st.title("Potable Water Predictor 🚰")
st.divider()

model = pickle.load(open('model.pkl','rb'))

def wat_pred(params):
    ip = np.array(params).reshape(1,-1)
    result = model.predict(ip)
    return result

def main():
    ph = st.number_input("**pH**",min_value=0.0,max_value=14.0,step=0.1,value=7.0)
    hardness = st.number_input("**Hardness**")
    solids = st.number_input("**Solids**")
    chloramines = st.number_input("**Chloramines**")
    Sulfate = st.number_input("**Sulfate**")
    conductivity = st.number_input("**Conductivity**")
    organic = st.number_input("**Organic Carbon**")
    Trihalomethanes = st.number_input("**Trihalomethanes**")
    turbidity = st.number_input("**Turbidity**",max_value=7)

    if st.button("**Calculate**"):
        try:
            params = (ph,hardness,solids,chloramines,Sulfate,conductivity,
                      organic,Trihalomethanes,turbidity)
            prediction = wat_pred(params)
            if prediction == 1:
                st.success("Your water is potable",icon="💧")
                st.snow()
            else:
                st.error("Your water is not potable. Do not consume!", icon = "🚫")

        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()