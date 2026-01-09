import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# ======================================
# LOAD DATASET
# ======================================
@st.cache_data
def load_instance(file):
    return pd.read_csv(file)

# ======================================
# ANT COLONY OPTIMIZATION (ACO)
# ======================================
class AntColonyKnapsack:
    def __init__(self, values, w1, w2, capacity,
                 n_ants=30, n_iterations=50,
                 alpha=1.0, beta=2.0, rho=0.1):

        self.values = values
        self.w1 = w1
        self.w2 = w2
        self.capacity = capacity
        self.n_items = len(values)

        self.n_ants = n_ants
        self.n_iterations = n_iterations
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
        all_solutions = []

        for _ in range(self.n_iterations):
            iteration_solutions = []

            for _ in range(self.n_ants):
                sol = self.construct_solution()
                result = self.evaluate(sol)

                if result is not None:
                    value, w2 = result
                    iteration_solutions.append((sol, value))
                    all_solutions.append([value, w2])

            self.update_pheromone(iteration_solutions)

        return np.array(all_solutions)

# ======================================
# PARETO FRONT FUNCTION
# ======================================
def pareto_front(points):
    pareto = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if (q[0] >= p[0] and q[1] <= p[1]) and (q[0] > p[0] or q[1] < p[1]):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return np.array(pareto)

# ======================================
# STREAMLIT UI
# ======================================
st.set_page_config(page_title="ACO Knapsack", layout="centered")
st.title("🐜 Ant Colony Optimization (ACO)")
st.subheader("Multi-Objective Knapsack Problem")

uploaded_file = st.file_uploader(
    "Upload Knapsack Instance (CSV)", type="csv"
)

capacity = st.slider("w1 Capacity (Constraint)", 100, 5000, 1000)

col1, col2 = st.columns(2)
with col1:
    n_ants = st.slider("Number of Ants", 10, 100, 30)
with col2:
    n_iter = st.slider("Iterations", 10, 200, 50)

if uploaded_file is not None:
    df = load_instance(uploaded_file)

    st.write("📄 Dataset Preview")
    st.dataframe(df.head())

    values = df["value"].values
    w1 = df["w1"].values
    w2 = df["w2"].values

    if st.button("🚀 Run ACO"):
        with st.spinner("Running Ant Colony Optimization..."):
            aco = AntColonyKnapsack(
                values, w1, w2,
                capacity=capacity,
                n_ants=n_ants,
                n_iterations=n_iter
            )

            results = aco.run()
            pareto = pareto_front(results)

        st.success("Optimization Completed!")

        st.write("### 📊 Pareto Front Result")

        fig, ax = plt.subplots()
        ax.scatter(results[:, 1], results[:, 0], alpha=0.3, label="All Solutions")
        ax.scatter(pareto[:, 1], pareto[:, 0], color="red", label="Pareto Front")
        ax.set_xlabel("Total w2 (Minimize)")
        ax.set_ylabel("Total Value (Maximize)")
        ax.legend()

        st.pyplot(fig)

        st.write("### 🔍 Pareto Solutions (Value, w2)")
        st.dataframe(pd.DataFrame(pareto, columns=["Value", "w2"]))

else:
    st.info("Please upload a knapsack instance CSV file.")
