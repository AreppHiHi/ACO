import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ===============================
# ANT COLONY OPTIMIZATION (ACO)
# ===============================
class ACO_Knapsack:
    def __init__(self, values, w1, w2, capacity,
                 n_ants=25, n_iter=40,
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
        results = []

        for _ in range(self.n_iter):
            iteration_solutions = []

            for _ in range(self.n_ants):
                sol = self.construct_solution()
                res = self.evaluate(sol)

                if res is not None:
                    value, w2 = res
                    iteration_solutions.append((sol, value))
                    results.append([value, w2])

            self.update_pheromone(iteration_solutions)

        return np.array(results)

# ===============================
# PARETO FRONT
# ===============================
def pareto_front(points):
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if (q[0] >= p[0] and q[1] <= p[1]) and (q[0] > p[0] or q[1] < p[1]):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return np.array(pareto)

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Knapsack Capacity Effect", layout="centered")

st.title("🎒 Knapsack Problem: Effect of Capacity")
st.markdown("""
This demo shows **how capacity (w1 constraint)** affects  
the **Pareto Front** in a multi-objective knapsack problem.
""")

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)

    values = df["value"].values
    w1 = df["w1"].values
    w2 = df["w2"].values

    total_w1 = np.sum(w1)

    st.subheader("🔧 Capacity Setting")
    ratio = st.slider(
        "Capacity Ratio (% of total w1)",
        min_value=0.1,
        max_value=0.6,
        value=0.3,
        step=0.05
    )

    capacity = int(ratio * total_w1)

    st.info(f"Total w1 = {int(total_w1)} | Capacity = {capacity}")

    if st.button("🚀 Run ACO"):
        with st.spinner("Optimizing with ACO..."):
            aco = ACO_Knapsack(
                values, w1, w2,
                capacity=capacity
            )

            results = aco.run()
            pareto = pareto_front(results)

        st.success("Optimization Completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Solutions Found", len(results))
        with col2:
            st.metric("Pareto Solutions", len(pareto))

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(results[:,1], results[:,0], alpha=0.3, label="All Solutions")
        ax.scatter(pareto[:,1], pareto[:,0], color="red", label="Pareto Front")
        ax.set_xlabel("Total w2 (Minimize)")
        ax.set_ylabel("Total Value (Maximize)")
        ax.set_title(f"Capacity = {int(ratio*100)}% of Total w1")
        ax.legend()

        st.pyplot(fig)

        st.subheader("📌 Interpretation")
        st.markdown(f"""
- Capacity is set to **{int(ratio*100)}%** of total w1  
- Smaller capacity → fewer items → lower value but lower w2  
- Larger capacity → more items → higher value but higher w2  
- Red points show **Pareto-optimal solutions**
        """)

else:
    st.warning("Please upload a knapsack dataset CSV file.")
