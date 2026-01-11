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
        """Individual ant builds a solution step-by-step."""
        solution = np.zeros(self.n_items)
        total_w1 = 0

        items = list(range(self.n_items))
        random.shuffle(items)

        for i in items:
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
        """Calculates total Value, w1 usage, and w2 usage."""
        total_value = np.sum(solution * self.values)
        total_w1 = np.sum(solution * self.w1)
        total_w2 = np.sum(solution * self.w2)

        if total_w1 > self.capacity:
            return None

        return total_value, total_w2

    def update_pheromone(self, solutions):
        """Evaporate and deposit pheromones based on solution quality."""
        self.pheromone *= (1 - self.rho)
        for sol, value in solutions:
            # More value = higher pheromone deposit
            self.pheromone += sol * (value / (np.max(self.values) + 1))

    def run(self):
        """Main loop for the ACO algorithm."""
        all_results = [] 

        for _ in range(self.n_iter):
            iteration_solutions = []
            for _ in range(self.n_ants):
                sol = self.construct_solution()
                res = self.evaluate(sol)

                if res is not None:
                    value, w2 = res
                    iteration_solutions.append((sol, value))
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
            if (q['value'] >= p['value'] and q['w2'] <= p['w2']) and \
               (q['value'] > p['value'] or q['w2'] < p['w2']):
                dominated = True
                break
        if not dominated:
            if not any(np.array_equal(p['mask'], x['mask']) for x in pareto):
                pareto.append(p)
    
    # Sort by value for better table presentation
    pareto.sort(key=lambda x: x['value'])
    return pareto

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="ACO Knapsack Optimizer", layout="wide")

st.title(" ACO Multi-Objective Knapsack Optimizer")
st.write("Balancing **Maximum Value** and **Minimum w2** under **w1** capacity constraints.")

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    
    # Dataset Preview Section
    st.subheader(" Dataset Preview")
    st.write(f"The dataset contains **{len(df)}** items.")
    st.dataframe(df.head(10)) 
    st.divider()
    
    # Sidebar Controls
    st.sidebar.subheader("Algorithm Configuration")
    n_ants = st.sidebar.slider("Number of Ants", 10, 100, 30)
    n_iter = st.sidebar.slider("Iterations", 10, 200, 50)
    
    values = df["value"].values
    w1 = df["w1"].values
    w2 = df["w2"].values
    total_w1_all = np.sum(w1)

    st.sidebar.subheader(" Capacity Setting (Constraint)")
    ratio = st.sidebar.slider("Capacity Ratio (w1)", 0.1, 0.9, 0.3)
    capacity = int(ratio * total_w1_all)
    st.sidebar.info(f"w1 Capacity Limit: {capacity}")

    # Initialize Session State to prevent data loss on widget interaction
    if 'aco_history' not in st.session_state:
        st.session_state.aco_history = None
    if 'pareto_results' not in st.session_state:
        st.session_state.pareto_results = None

    if st.button(" Run ACO Optimization"):
        with st.spinner("Ants are searching for optimal solutions..."):
            aco = ACO_Knapsack(values, w1, w2, capacity, n_ants=n_ants, n_iter=n_iter)
            all_history = aco.run()
            pareto_list = get_pareto_front(all_history)
            
            # Save results to session state
            st.session_state.aco_history = all_history
            st.session_state.pareto_results = pareto_list
            
        st.success(f"Success! Found {len(pareto_list)} Pareto-optimal solutions.")

    # Display results if data exists in Session State
    if st.session_state.pareto_results is not None:
        all_history = st.session_state.aco_history
        pareto_list = st.session_state.pareto_results
        
        df_all = pd.DataFrame(all_history)
        df_pareto = pd.DataFrame(pareto_list)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Pareto Front Visualization")
            fig, ax = plt.subplots()
            ax.scatter(df_all['w2'], df_all['value'], color='grey', alpha=0.2, label="Valid Solutions")
            ax.scatter(df_pareto['w2'], df_pareto['value'], color='red', s=50, label="Pareto Front")
            ax.set_xlabel("Total w2 (Minimize)")
            ax.set_ylabel("Total Value (Maximize)")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Pareto Table (Value vs w2)")
            st.dataframe(df_pareto[['value', 'w2']])

        st.divider()
        
        # ITEM INSPECTION SECTION
        st.subheader(" Inspect Selected Items")
        
        # Formatting function for the selectbox labels
        def format_solution_label(idx):
            sol = df_pareto.iloc[idx]
            n_items = int(np.sum(sol['mask']))
            return f"Solution {idx}: [Value: {sol['value']}] | [w2: {sol['w2']}] | [{n_items} Items Selected]"

        selected_idx = st.selectbox(
            "Select a solution index to see the item breakdown:",
            options=range(len(df_pareto)),
            format_func=format_solution_label,
            key="inspect_selector"
        )
        
        # Retrieve the mask for the selected solution
        chosen_mask = df_pareto.iloc[selected_idx]['mask']
        selected_indices = np.where(chosen_mask == 1)[0]
        selected_items_table = df.iloc[selected_indices]
        
        st.write(f"Displaying **{len(selected_items_table)} items** chosen for **Solution {selected_idx}**:")
        st.dataframe(selected_items_table)
        
        # Summary Analysis Info Box
        st.info(f"""
        **Analysis for Solution {selected_idx}:**
        * Total Profit (Value): **{selected_items_table['value'].sum()}**
        * Total Operational Cost (w2): **{selected_items_table['w2'].sum()}**
        * Total Items Selected: **{len(selected_items_table)}**
        * w1 Capacity Usage: **{selected_items_table['w1'].sum()} / {capacity}**
        """)

else:
    st.warning("Please upload a CSV dataset to begin.")
