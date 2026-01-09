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
                # Heuristik: Suka Value tinggi & w2 rendah
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
        all_results = [] # Menyimpan sejarah semua solusi yang sah

        for _ in range(self.n_iter):
            iteration_solutions = []
            for _ in range(self.n_ants):
                sol = self.construct_solution()
                res = self.evaluate(sol)

                if res is not None:
                    value, w2 = res
                    iteration_solutions.append((sol, value))
                    # Simpan maklumat lengkap termasuk mask binari
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
    pareto = []
    for p in all_data:
        dominated = False
        for q in all_data:
            # Objektif: Maximize Value (index 0), Minimize w2 (index 1)
            if (q['value'] >= p['value'] and q['w2'] <= p['w2']) and \
               (q['value'] > p['value'] or q['w2'] < p['w2']):
                dominated = True
                break
        if not dominated:
            # Elakkan simpan solusi yang mempunyai mask yang sama
            if not any(np.array_equal(p['mask'], x['mask']) for x in pareto):
                pareto.append(p)
    
    # Sort ikut value untuk paparan cantik
    pareto.sort(key=lambda x: x['value'])
    return pareto

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="ACO Knapsack Optimizer", layout="wide")

st.title("🎒 ACO Multi-Objective Knapsack")
st.write("Mencari keseimbangan antara **Maximum Value** dan **Minimum w2** dengan had **w1**.")

uploaded_file = st.sidebar.file_uploader("Muat Naik Dataset (CSV)", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    
    # Sidebar Controls
    st.sidebar.subheader("Konfigurasi Algoritma")
    n_ants = st.sidebar.slider("Bilangan Semut", 10, 100, 30)
    n_iter = st.sidebar.slider("Iterasi", 10, 200, 50)
    
    values = df["value"].values
    w1 = df["w1"].values
    w2 = df["w2"].values
    total_w1_all = np.sum(w1)

    st.sidebar.subheader("🔧 Had Kapasiti (Constraint)")
    ratio = st.sidebar.slider("Nisbah Kapasiti (w1)", 0.1, 0.9, 0.3)
    capacity = int(ratio * total_w1_all)
    st.sidebar.info(f"Had w1: {capacity}")

    if st.button("🚀 Jalankan Optimasi ACO"):
        with st.spinner("Semut sedang mencari jalan terbaik..."):
            aco = ACO_Knapsack(values, w1, w2, capacity, n_ants=n_ants, n_iter=n_iter)
            all_history = aco.run()
            pareto_list = get_pareto_front(all_history)
            
        # Paparan Keputusan
        st.success(f"Selesai! Berjaya menemui {len(pareto_list)} solusi Pareto.")
        
        col1, col2 = st.columns([2, 1])
        
        df_all = pd.DataFrame(all_history)
        df_pareto = pd.DataFrame(pareto_list)

        with col1:
            st.subheader("Visualisasi Pareto Front")
            fig, ax = plt.subplots()
            ax.scatter(df_all['w2'], df_all['value'], color='grey', alpha=0.2, label="Semua Solusi")
            ax.scatter(df_pareto['w2'], df_pareto['value'], color='red', s=50, label="Pareto Front")
            ax.set_xlabel("Total w2 (Kos - Lagi sikit lagi bagus)")
            ax.set_ylabel("Total Value (Untung - Lagi banyak lagi bagus)")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Jadual Pareto (Value, w2)")
            # Papar jadual tanpa kolum mask untuk kebersihan
            st.dataframe(df_pareto[['value', 'w2']])

        st.divider()
        
        # BAHAGIAN SEMAK ITEM
        st.subheader("🔍 Semak Item dalam Solusi")
        selected_idx = st.selectbox(
            "Pilih satu solusi untuk melihat senarai item di dalamnya:",
            options=range(len(df_pareto)),
            format_func=lambda x: f"Solusi {x}: Value={df_pareto.iloc[x]['value']}, w2={df_pareto.iloc[x]['w2']}"
        )
        
        # Dapatkan mask bagi baris yang dipilih
        chosen_mask = df_pareto.iloc[selected_idx]['mask']
        selected_indices = np.where(chosen_mask == 1)[0]
        
        # Paparkan item
        selected_items_table = df.iloc[selected_indices]
        st.write(f"Berikut adalah **{len(selected_items_table)} item** yang dipilih untuk Solusi {selected_idx}:")
        st.dataframe(selected_items_table)
        
        # Pengesahan Matematik
        st.info(f"""
        **Analisis Ringkas Solusi {selected_idx}:**
        * Total Value: **{selected_items_table['value'].sum()}**
        * Total w2: **{selected_items_table['w2'].sum()}**
        * Penggunaan Kapasiti w1: **{selected_items_table['w1'].sum()} / {capacity}**
        """)

else:
    st.warning("Sila muat naik fail CSV dataset untuk bermula.")
