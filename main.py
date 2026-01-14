import streamlit as st
import numpy as np
import pandas as pd
import random
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="ACO Knapsack Optimizer",
    layout="wide"
)

st.title("🐜 ACO Multi-Objective Knapsack Optimizer")
st.write(
    "Optimizing **Maximum Value** and **Minimum w2** under **w1 capacity constraint** "
    "using **Ant Colony Optimization (ACO)**."
)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# ===============================
# DATASET SELECTION (LOCAL CSV)
# ===============================
DATASET_FOLDER = "."

csv_files = sorted([f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")])

if not csv_files:
    st.error("No CSV datasets found in the project directory.")
    st.stop()

st.sidebar.subheader(" Dataset Selection")
selected_file = st.sidebar.selectbox(
    "Choose a dataset",
    options=csv_files
)

df = load_data(os.path.join(DATASET_FOLDER, selected_file))
st.sidebar.success(f"Loaded: {selected_file}")

# ===============================
# DATA PREVIEW
# ===============================
st.subheader("📄 Dataset Preview")
st.write(f"Total items: **{len(df)}**")
st.dataframe(df.head(10))
st.divider()

# ===============================
# ACO CLASS
# ===============================
class ACO_Knapsack:
    def __init__(self, values, w1, w2, capacity,
                 n_ants=30, n_iter=50,
                 alpha=1.0, beta=2.0, rho=0.1):

        self.values = values
        self.w1 = w1
        self.w2 = w2
        self.capacity = capacity
        self.n_items = len(values)

        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.pheromone = np.ones(self.n_items)

    def construct_solution(self):
        solution = np.zeros(self.n_items)
        total_w1 = 0

        items = list(range(self.n_items))
        random.shuffle(items)

        for i in items:
            if total_w1 + self.w1[i] <= self.capacity:
                heuristic = self.values[i] / (self.w2[i] + 1)
                prob = (self.pheromone[i] ** self.alpha) * (heuristic ** self.beta)

                if random.random() < prob / (1 + prob):
                    solution[i] = 1
                    total_w1 += self.w1[i]

        return solution

    def evaluate(self, solution):
        total_value = np.sum(solution * self.values)
        total_w1 = np.sum(solution * self.w1)
        total_w2 = np.sum(solution * self.w2)

        if total_w1 > self.capacity:
            return None

        return total_value, total_w2

    def update_pheromone(self, solutions):
        self.pheromone *= (1 - self.rho)
        for sol, value in solutions:
            self.pheromone += sol * (value / (np.max(self.values) + 1))

    def run(self):
        all_results = []

        for _ in range(self.n_iter):
            iteration_solutions = []

            for _ in range(self.n_ants):
                sol = self.construct_solution()
                res = self.evaluate(sol)

                if res:
                    value, w2 = res
                    iteration_solutions.append((sol, value))
                    all_results.append({
                        "mask": sol,
                        "value": value,
                        "w2": w2
                    })

            self.update_pheromone(iteration_solutions)

        return all_results

# ===============================
# PARETO FRONT FUNCTION
# ===============================
def get_pareto_front(data):
    pareto = []

    for p in data:
        dominated = False
        for q in data:
            if (q["value"] >= p["value"] and q["w2"] <= p["w2"]) and \
               (q["value"] > p["value"] or q["w2"] < p["w2"]):
                dominated = True
                break
        if not dominated:
            if not any(np.array_equal(p["mask"], x["mask"]) for x in pareto):
                pareto.append(p)

    pareto.sort(key=lambda x: x["value"])
    return pareto

# ===============================
# SIDEBAR PARAMETERS
# ===============================
st.sidebar.subheader(" ACO Parameters")

n_ants = st.sidebar.slider("Number of Ants", 10, 100, 30)
n_iter = st.sidebar.slider("Iterations", 10, 200, 50)

values = df["value"].values
w1 = df["w1"].values
w2 = df["w2"].values

total_w1 = np.sum(w1)

st.sidebar.subheader(" Capacity Constraint")
ratio = st.sidebar.slider("Capacity Ratio (w1)", 0.1, 0.9, 0.3)
capacity = int(ratio * total_w1)
st.sidebar.info(f"w1 Capacity = {capacity}")

# ===============================
# SESSION STATE
# ===============================
if "history" not in st.session_state:
    st.session_state.history = None
if "pareto" not in st.session_state:
    st.session_state.pareto = None

# ===============================
# RUN OPTIMIZATION
# ===============================
if st.button(" Run ACO Optimization"):
    with st.spinner("Ants are searching for optimal solutions..."):
        aco = ACO_Knapsack(
            values, w1, w2, capacity,
            n_ants=n_ants,
            n_iter=n_iter
        )
        history = aco.run()
        pareto = get_pareto_front(history)

        st.session_state.history = history
        st.session_state.pareto = pareto

    st.success(f"Found {len(pareto)} Pareto-optimal solutions.")

# ===============================
# RESULTS VISUALIZATION
# ===============================
if st.session_state.pareto:

    history = st.session_state.history
    pareto = st.session_state.pareto

    df_all = pd.DataFrame(history)
    df_pareto = pd.DataFrame(pareto)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(" Pareto Front Visualization")
        fig, ax = plt.subplots()
        ax.scatter(df_all["w2"], df_all["value"], alpha=0.2, label="All Solutions")
        ax.scatter(df_pareto["w2"], df_pareto["value"], color="red", s=50, label="Pareto Front")
        ax.set_xlabel("Total w2 (Minimize)")
        ax.set_ylabel("Total Value (Maximize)")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("📊 Pareto Solutions")
        st.dataframe(df_pareto[["value", "w2"]])

    st.divider()

    # ===============================
    # SOLUTION INSPECTION
    # ===============================
    st.subheader("🔍 Inspect Selected Solution")

    def format_label(i):
        sol = df_pareto.iloc[i]
        n_items = int(np.sum(sol["mask"]))
        return f"Solution {i} | Value: {sol['value']} | w2: {sol['w2']} | Items: {n_items}"

    idx = st.selectbox(
        "Choose a Pareto-optimal solution",
        options=range(len(df_pareto)),
        format_func=format_label
    )

    mask = df_pareto.iloc[idx]["mask"]
    selected_items = df.iloc[np.where(mask == 1)[0]]

    st.dataframe(selected_items)

    st.info(f"""
    **Solution Summary**
    - Total Value: {selected_items["value"].sum()}
    - Total w2: {selected_items["w2"].sum()}
    - Total w1 Used: {selected_items["w1"].sum()} / {capacity}
    - Items Selected: {len(selected_items)}
    """)
