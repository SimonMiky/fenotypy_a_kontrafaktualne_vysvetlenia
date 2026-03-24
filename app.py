import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


st.set_page_config(layout="wide")

# nastavenie tlacidiel
st.markdown("""
<style>

/* Farba tlačidla */
div.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    height: 50px;
    font-size: 16px;
    font-weight: 600;
}

/* Hover efekt */
div.stButton > button:hover {
    background-color: #4e8cd9;
    color: white;
}

/* Medzera nad tlačidlom */
div.stButton {
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# nastavenie pozadia
st.markdown("""
<style>
.stApp {
    background-color: #4a5f80;
}
</style>
""", unsafe_allow_html=True)

# zarovnanie nadpisov
st.markdown("""
<style>
h1, h2, h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

#max sirka aplikacie
st.markdown("""
<style>
.block-container {
    max-width: 1400px;
}
</style>
""", unsafe_allow_html=True)

# nastavenie infoboxov
def info_box(text):
    st.markdown(f"""
    <div style="
        background-color: #4e8cd9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 6px solid #1f77b4;
        color: #ffffff;
        font-size: 16px;
        margin-bottom: 1rem;
    ">
        {text}
    </div>
    """, unsafe_allow_html=True)

# ==========================
# NAČÍTANIE MODELU
# ==========================

model = joblib.load("kmeans_model_4vlna15_.pkl")
scaler = joblib.load("scaler_4vlna15.pkl")
X_scaled = pd.read_pickle("X_scaled_4vlna15.pkl")
X_orig = pd.read_pickle("X_orig_4vlna15.pkl")

centroids = model.cluster_centers_

# POTREBNE NASTAVIT ZAVAZNOSTI ZHLUKOV
cluster_severity = {
    0: "Najmenej závažný",
    1: "Stredne závažný",
    2: "Najzávažnejší"
}

st.title("Fenotypy a kontrafaktuálne vysvetlenia pacientov")

# pocitanie medianov 
cluster_medians = (
    X_orig
    .assign(cluster=model.labels_)
    .groupby("cluster")
    .median()
)

# Tabulka medianov fenotypov 
def plot_cluster_heatmap(cluster_medians, highlight_cluster=None):

    fig, ax = plt.subplots(figsize=(5, 5.6))

    sns.heatmap(
        cluster_medians.T,
        cmap="coolwarm",
        norm=LogNorm(),
        annot=True,
        ax=ax,
        fmt=".2f",
        linewidths=1,
        linecolor='gray'
    )

    ax.set_title("Mediány atribútov podľa fenotypov")
    ax.set_ylabel("Atribút")
    ax.set_xlabel("Fenotyp")

    return fig

# Radarovy graf standardizovanych hodn;t pacienta a jeho fenotypu
def plot_radar_scaled(centroids, patient_scaled_df, cluster, feature_names):

    categories = feature_names
    N = len(categories)

    # centroid vybraného fenotypu (už standardizovany)
    cluster_vals = centroids[cluster]
    patient_vals = patient_scaled_df.values.flatten()

    cluster_vals = cluster_vals.tolist()
    patient_vals = patient_vals.tolist()

    cluster_vals += cluster_vals[:1]
    patient_vals += patient_vals[:1]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(111, polar=True)

    ax.set_ylim(-3, 3)   # rozsah SD grafu
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(axis='y', labelsize=7, colors="gray")

    ax.plot(angles, cluster_vals, linewidth=1.5, label=(f"Fenotyp {cluster}"))
    ax.fill(angles, cluster_vals, alpha=0.2)

    ax.plot(angles, patient_vals, linewidth=1.5, linestyle="dashed", label="Pacient")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7)

    ax.set_title("Porovnanie pacienta a jeho fenotypu", pad=30)

    ax.legend(loc="lower right", fontsize = 9, bbox_to_anchor=(1.2, -0.1))

    ax.spines["polar"].set_visible(False)
    ax.grid(alpha=0.7)
    
    return fig

# ==========================
# INPUT PACIENTA
# ==========================

st.markdown(f"""
    <div style="
        background-color: #4e8cd9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0px solid #1f77b4;
        color: #ffffff;
        font-size: 16px;
        margin-bottom: 1rem;
    ">
        Táto aplikácia bola vytvorená v rámci diplomovej práce a slúži na analýzu fenotypov pacientov pomocou metód zhlukovania a kontrafaktuálnych vysvetlení. Jej cieľom je priradenie nového pacienta do modelu a lepšie porozumenie rozdielov medzi jednotlivými fenotypmi. \nAplikácia je rozdelená na dve hlavné časti. V prvej časti používateľ zadáva údaje o novom pacientovi, ktorý je následne automaticky priradený do jedného z fenotypov. Súčasťou tejto časti je aj vizualizácia hodnôt pacienta v porovnaní s jeho fenotypom a tiež porovnanie fenotypov navzájom. \n Druhá časť aplikácie je zameraná na kontrafaktuálne vysvetlenia. Tu si môže používateľ zvoliť cieľový fenotyp pacienta a aplikácia následne určí minimálne zmeny vo vstupných atribútoch, ktoré by viedli k preradeniu nového pacienta do zvoleného fenotypu. Týmto spôsobom aplikácia poskytuje hlbší pohľad do faktorov ovplyvňujúcich zaradenie pacienta.
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.subheader("1. časť - Fenotypyzácia nového pacienta")

info_box("Pre priradenie nového pacienta do existujúceho fenotypu zadajte hodnoty atribútov pacienta. Hodnoty sú predvyplnené mediánmi hodnôt atribútov. Po priradení bude pacient zaradený do jedného z troch zhlukov, teda fenotypov, od najmenej závažného po najzávažnejší fenotyp.")

st.markdown(
        "<p style='font-size:20px; font-weight:600;'>Zadajte údaje pacienta:",
        unsafe_allow_html=True
    )

feature_names = X_orig.columns.tolist()

feature_units = {
    "NE/LY(NLR) last": "",
    "S-CRP last": "mg/l",
    "S-IL6 last": "ng/l",
    "S-Alb last": "g/l",
    "S-Na last": "mmol/l",
    "S-Urea last": "mmol/l",
    "D-dimér HS last": "mg/l",
    "S-PBNP last": "ng/l",
    "S-CL last": "mmol/l",
    "SatO2 %": "%",
    "P-Laktát last": "mmol/l",
    "Fib last": "g/l",
    "S-Gluk last": "mmol/l",
    "WBC last": "10^9/l",
    "Vek": "roky",
}

input_data = {}

n_cols = 5                  # polia na zadanie hodnot pacienta v 5 stlpcoch
cols = st.columns(n_cols)



for i, feature in enumerate(feature_names):
    col = cols[i % n_cols]
    with col:
        unit = feature_units.get(feature, "")     #jednotky
        
        input_data[feature] = st.number_input(
            f"{feature} ({unit})" if unit else feature,
            value=float(X_orig[feature].median())         # predvyplnen8 hodnota je medi8n
        )

# convert na DataFrame
new_patient_df = pd.DataFrame([input_data])

# ==========================
# INICIALIZACIA
# ==========================

if "cluster" not in st.session_state:
    st.session_state.cluster = None

if "new_patient_scaled_df" not in st.session_state:
    st.session_state.new_patient_scaled_df = None

if "new_patient_df" not in st.session_state:
    st.session_state.new_patient_df = None


# ==========================
# PREDIKCIA ZHLUKU
# ==========================

with cols[2]:
    pressed = st.button("Priradiť pacienta", use_container_width=True)

    if pressed:

        new_patient_df = pd.DataFrame([input_data])
        new_patient_df = new_patient_df[X_orig.columns]

        new_patient_scaled = scaler.transform(new_patient_df)

        new_patient_scaled_df = pd.DataFrame(
            new_patient_scaled,
            columns=X_orig.columns
        )

        cluster = model.predict(new_patient_scaled_df)[0]

        # ULOŽÍME 
        st.session_state.cluster = cluster
        st.session_state.new_patient_scaled_df = new_patient_scaled_df
        st.session_state.new_patient_df = new_patient_df


# ==========================
# ZOBRAZENIE VÝSLEDKU Radaroveho grafu a tabulky medianov
# ==========================

if st.session_state.cluster is not None:

    cluster = st.session_state.cluster

    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
            margin-top: 2rem;
        ">
            <div style="
                background-color: #57ba57; color: white; padding: 12px 25px; border-radius: 8px; font-size: 18px; font-weight: 500; text-align: center;
            ">
                Pacient patrí do fenotypu: {cluster} – {cluster_severity[cluster]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    info_box(f"Graf naľavo zobrazuje porovnanie štandardizovaných hodnôt pacienta a centroidu jeho fenotypu, teda fenotypu {cluster}. Tabuľka napravo zobrazuje mediány hodnôt atribútov jednotlivých fenotypov.")

    left_panel, right_panel = st.columns([1,1])

    with left_panel:
        radar_fig = plot_radar_scaled(
            centroids,
            st.session_state.new_patient_scaled_df,
            cluster,
            feature_names
        )
        st.pyplot(radar_fig, use_container_width=False)

    with right_panel:
        heatmap_fig = plot_cluster_heatmap(cluster_medians)
        st.pyplot(heatmap_fig, use_container_width=False)

    possible_targets = [c for c in cluster_severity.keys() if c != cluster]

    st.divider()

    st.subheader("2. časť - Kontrafaktuálne vysvetlenia")

    info_box(f"Kontrafaktuálne vysvetlenia popisujú aké minimálne zmeny by museli u daného pacienta nastať aby patril do iného fenotypu. Zo zoznamu vyberte fenotyp, pre ktorý chcete tieto minimálne zmeny určiť. Výsledkom budú minimálne zmeny vzhľadom ku najbližšiemu pacientovi (kontrafaktuálny pacient) vo vybranom fenotype.")

    st.markdown(
        "<p style='font-size:20px; font-weight:600;margin-bottom:-40px;'>Vyberte cieľový fenotyp:",
        unsafe_allow_html=True
    )

    desired_cluster = st.selectbox(
        "",
        possible_targets,
        format_func=lambda x: f"{x} – {cluster_severity[x]}"
    )

    # ==========================
    # KONTRAFAKTUÁL - Vypocet a zobrazenie
    # ==========================
    cf_left, cf_center, cf_right = st.columns([2,3,2])

    with cf_center:
        pressed2 = st.button("Vypočítať kontrafaktuálne vysvetlenia", use_container_width=True)
        
        if pressed2:

            new_patient_scaled_df = st.session_state.new_patient_scaled_df
            new_patient_df = st.session_state.new_patient_df

            mask = model.labels_ == desired_cluster
            X_target_scaled = X_scaled[mask]

            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(X_target_scaled.values)

            dist, neighbor_pos = nn.kneighbors(new_patient_scaled_df.values)
            anchor_idx = X_target_scaled.index[neighbor_pos[0][0]]

            x_anchor_scaled = X_scaled.loc[anchor_idx].values
            x_patient_scaled = new_patient_scaled_df.values.ravel()

            deltas = x_anchor_scaled - x_patient_scaled
            order = np.argsort(-np.abs(deltas))

            x_cf_scaled = x_patient_scaled.copy()
            changed = []

            for j in order:
                x_cf_scaled[j] = x_anchor_scaled[j]
                changed.append(feature_names[j])

                new_cluster = np.argmin(
                    np.linalg.norm(centroids - x_cf_scaled, axis=1)
                )

                if new_cluster == desired_cluster:
                    break

            x_anchor_orig = X_orig.loc[anchor_idx]
            x_cf_orig = new_patient_df.iloc[0].copy()

            for col in changed:
                x_cf_orig[col] = x_anchor_orig[col]

            delta_orig = x_cf_orig - new_patient_df.iloc[0]

            cf_table = pd.DataFrame({
                "Nový pacient": new_patient_df.iloc[0],
                "Kontrafaktuálny pacient": x_cf_orig,
                "Zmena": delta_orig
            })

            cf_table = cf_table.loc[changed].sort_values(
                by="Zmena",
                key=lambda s: s.abs(),
                ascending=False
            )

            with cf_center:
                st.markdown("##### Minimálne zmeny pre preradenie pacienta:")
                st.dataframe(cf_table, use_container_width=True)
