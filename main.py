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
    """Memuatkan dataset CSV."""
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

        # Pheromone trail untuk setiap item
        self.pheromone = np.ones(self.n_items)

    def construct_solution(self):
        """Semut membina solusi secara berperingkat."""
        solution = np.zeros(self.n_items)
        total_w1 = 0

        items = list(range(self.n_items))
        random.shuffle(items)

        for i in items:
            if total_w1 + self.w1[i] <= self.capacity:
                # Heuristic: Utamakan Value tinggi dan w2 rendah
                heuristic = self.values[i] / (self.w2[i] + 1)
                
                # Formula kebarangkalian berdasarkan Pheromone dan Heuristic
                prob = (self.pheromone[i] ** self.alpha) * (heuristic ** self.beta)

                if random.random() < prob / (1 + prob):
                    solution[i] = 1
                    total_w1 += self.w1[i]

        return solution

    def evaluate(self, solution):
        """Kira jumlah Value, w1, dan w2."""
        total_value = np.sum(solution * self.values)
        total_w1 = np.sum(solution * self.w1)
        total_w2 = np.sum(solution * self.w2)

        if total_w1 > self.capacity:
            return None

        return total_value, total_w2

    def update_pheromone(self, solutions):
        """Proses penyejatan dan penambahan pheromone."""
        self.pheromone *= (1 - self.rho)
        for sol, value in solutions:
            self.pheromone += sol * (value / (np.max(self.values) + 1))

    def run(self):
        """Loop utama algoritma ACO."""
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
    """Menapis solusi untuk mencari Pareto Front (Non-dominated)."""
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
    
    pareto.sort(key=lambda x: x['value'])
    return pareto

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="ACO Knapsack Optimizer", layout="wide")

st.title(" ACO Multi-Objective Knapsack")
st.write("Mencari keseimbangan antara **Maksimum Value** dan **Minimum w2** di bawah kekangan **w1**.")

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    
    # Dataset Preview
    st.subheader("📊 Dataset Preview")
    st.write(f"Dataset mengandungi **{len(df)}** item.")
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
    st.sidebar.info(f"w1 Limit (Capacity): {capacity}")

    # Inisialisasi Session State
    if 'aco_history' not in st.session_state:
        st.session_state.aco_history = None
    if 'pareto_results' not in st.session_state:
        st.session_state.pareto_results = None

    if st.button(" Run ACO Optimization"):
        with st.spinner("Semut sedang mencari jalan optimum..."):
            aco = ACO_Knapsack(values, w1, w2, capacity, n_ants=n_ants, n_iter=n_iter)
            all_history = aco.run()
            pareto_list = get_pareto_front(all_history)
            
            # Simpan ke session state supaya tidak hilang bila guna selectbox
            st.session_state.aco_history = all_history
            st.session_state.pareto_results = pareto_list
            
        st.success(f"Berjaya! Menemui {len(pareto_list)} solusi Pareto.")

    # Paparkan hasil jika data wujud dalam Session State
    if st.session_state.pareto_results is not None:
        all_history = st.session_state.aco_history
        pareto_list = st.session_state.pareto_results
        
        df_all = pd.DataFrame(all_history)
        df_pareto = pd.DataFrame(pareto_list)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Pareto Front Visualization")
            fig, ax = plt.subplots()
            ax.scatter(df_all['w2'], df_all['value'], color='grey', alpha=0.2, label="Semua Solusi Sah")
            ax.scatter(df_pareto['w2'], df_pareto['value'], color='red', s=50, label="Pareto Front")
            ax.set_xlabel("Total w2 (Minimumkan)")
            ax.set_ylabel("Total Value (Maksimumkan)")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Pareto Table (Value vs w2)")
            st.dataframe(df_pareto[['value', 'w2']])

        st.divider()
        
        # ITEM INSPECTION SECTION
        st.subheader("🔍 Inspect Selected Items")
        
        # Fungsi label untuk selectbox
        def format_solusi(idx):
            sol = df_pareto.iloc[idx]
            n_items = int(np.sum(sol['mask']))
            return f"Solution {idx}: [Value: {sol['value']}] | [w2: {sol['w2']}] | [{n_items} Items]"

        selected_idx = st.selectbox(
            "Pilih indeks solusi untuk melihat senarai item:",
            options=range(len(df_pareto)),
            format_func=format_solusi,
            key="inspect_selector"
        )
        
        # Ambil mask bagi baris yang dipilih
        chosen_mask = df_pareto.iloc[selected_idx]['mask']
        selected_indices = np.where(chosen_mask == 1)[0]
        selected_items_table = df.iloc[selected_indices]
        
        st.write(f"Memaparkan **{len(selected_items_table)} item** yang dipilih untuk **Solution {selected_idx}**:")
        st.dataframe(selected_items_table)
        
        # Analisis Ringkas
        st.info(f"""
        **Analisis Solution {selected_idx}:**
        * Jumlah Keuntungan (Value): **{selected_items_table['value'].sum()}**
        * Jumlah Kos (w2): **{selected_items_table['w2'].sum()}**
        * Bilangan Item: **{len(selected_items_table)}**
        * Penggunaan Kapasiti w1: **{selected_items_table['w1'].sum()} / {capacity}**
        """)

else:
    st.warning("Sila muat naik dataset CSV untuk bermula.")
