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
    """Loads the CSV dataset."""
    return pd.read_csv(file)

# ===============================
# ANT COLONY OPTIMIZATION (ACO)
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

        # Pheromone trail for each item
        self.pheromone = np.ones(self.n_items)

    def construct_solution(self):
        """Individual ant builds a solution."""
        solution = np.zeros(self.n_items)
        total_w1 = 0

        # Randomize item order to explore different paths
        items = list(range(self.n_items))
        random.shuffle(items)

        for i in items:
            # Constraint check: Does the item fit in the capacity?
            if total_w1 + self.w1[i] <= self.capacity:
                # Heuristic: Favor high Value and low w2
                heuristic = self.values[i] / (self.w2[i] + 1)
                
                # Probability formula based on Pheromone and Heuristic
                prob = (self.pheromone[i] ** self.alpha) * (heuristic ** self.beta)

                if random.random() < prob / (1 + prob):
                    solution[i] = 1
                    total_w1 += self.w1[i]

        return solution

    def evaluate(self, solution):
        """Calculates total value, w1 usage, and w2 usage."""
        total_value = np.sum(solution * self.values)
        total_w1 = np.sum(solution * self.w1)
        total_w2 = np.sum(solution * self.w2)

        # Final check if the solution is valid
        if total_w1 > self.capacity:
            return None

        return total_value, total_w2

    def update_pheromone(self, solutions):
        """Evaporate and deposit pheromones based on solution quality."""
        self.pheromone *= (1 - self.rho)
        for sol, value in solutions:
            # More value = more pheromone deposit
            self.pheromone += sol * (value / (np.max(self.values) + 1))

    def run(self):
        """Main loop for the ACO algorithm."""
        all_results = [] # Stores history of all valid solutions

        for _ in range(self.n_iter):
            iteration_solutions = []
            for _ in range(self.n_ants):
                sol = self.construct_solution()
                res = self.evaluate(sol)

                if res is not None:
                    value, w2 = res
                    iteration_solutions.append((sol, value))
                    # Store mask along with objective values
                    all_results.append({
                        'mask': sol,
                        'value': value,
                        'w2': w2
                    })

            self.update_pheromone(iteration_solutions)

        return all_results

# ===============================
# PARETO FRONT
# ===============================
def get_pareto_front(all_data):
    """Filters solutions to find the non-dominated Pareto Front."""
    pareto = []
    for p in all_data:
        dominated = False
        for q in all_data:
            # p is dominated if q has higher value AND lower w2
            if (q['value'] >= p['value'] and q['w2'] <= p['w2']) and \
               (q['value'] > p['value'] or q['w2'] < p['w2']):
                dominated = True
                break
        if not dominated:
            # Avoid duplicate binary masks
            if not any(np.array_equal(p['mask'], x['mask']) for x in pareto):
                pareto.append(p)
    
    # Sort by value for better table presentation
    pareto.sort(key=lambda x: x['value'])
    return pareto

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="ACO Knapsack Optimizer", layout="wide")

st.title("🎒 ACO Multi-Objective Knapsack")
st.write("Finding the balance between **Maximum Value** and **Minimum w2** under **w1** constraints.")

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    
    # Sidebar Controls
    st.sidebar.subheader("Algorithm Configuration")
    n_ants = st.sidebar.slider("Number of Ants", 10, 100, 30)
    n_iter = st.sidebar.slider("Iterations", 10, 200, 50)
    
    values = df["value"].values
    w1 = df["w1"].values
    w2 = df["w2"].values
    total_w1_all = np.sum(w1)

    st.sidebar.subheader("🔧 Capacity Setting (Constraint)")
    ratio = st.sidebar.slider("Capacity Ratio (w1)", 0.1, 0.9, 0.3)
    capacity = int(ratio * total_w1_all)
    st.sidebar.info(f"w1 Limit: {capacity}")

    if st.button("🚀 Run ACO Optimization"):
        with st.spinner("Ants are searching for optimal paths..."):
            aco = ACO_Knapsack(values, w1, w2, capacity, n_ants=n_ants, n_iter=n_iter)
            all_history = aco.run()
            pareto_list = get_pareto_front(all_history)
            
        # Display Results
        st.success(f"Success! Found {len(pareto_list)} Pareto solutions.")
        
        col1, col2 = st.columns([2, 1])
        
        df_all = pd.DataFrame(all_history)
        df_pareto = pd.DataFrame(pareto_list)

        with col1:
            st.subheader("Pareto Front Visualization")
            fig, ax = plt.subplots()
            ax.scatter(df_all['w2'], df_all['value'], color='grey', alpha=0.2, label="All Valid Solutions")
            ax.scatter(df_pareto['w2'], df_pareto['value'], color='red', s=50, label="Pareto Front")
            ax.set_xlabel("Total w2 (Minimize)")
            ax.set_ylabel("Total Value (Maximize)")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Pareto Table (Value vs w2)")
            # Show table without the mask column for clarity
            st.dataframe(df_pareto[['value', 'w2']])

        st.divider()
        
        # ITEM INSPECTION SECTION
        st.subheader("🔍 Inspect Selected Items")
        selected_idx = st.selectbox(
            "Select a solution index to see its items:",
            options=range(len(df_pareto)),
            format_func=lambda x: f"Solution {x}: Value={df_pareto.iloc[x]['value']}, w2={df_pareto.iloc[x]['w2']}"
        )
        
        # Get the binary mask for the selected Pareto row
        chosen_mask = df_pareto.iloc[selected_idx]['mask']
        selected_indices = np.where(chosen_mask == 1)[0]
        
        # Filter the original dataframe to show selected items
        selected_items_table = df.iloc[selected_indices]
        st.write(f"Displaying **{len(selected_items_table)} items** chosen for Solution {selected_idx}:")
        st.dataframe(selected_items_table)
        
        # Mathematical Confirmation
        st.info(f"""
        **Solution {selected_idx} Analysis:**
        * Total Value: **{selected_items_table['value'].sum()}**
        * Total w2: **{selected_items_table['w2'].sum()}**
        * w1 Capacity Usage: **{selected_items_table['w1'].sum()} / {capacity}**
        """)

else:
    st.warning("Please upload a CSV dataset to begin.")
