
##################################################################
##### POPISNÉ MODELY PRE ANALÝZU DÁT O PACIENTOCH S COVID-19 #####
##################################################################

### Aplikácia pre fonotypizáciu pacienta a jeho kontrafaktuálnych vysvetlení, na základe k-means algoritmu štvrtej vlny pandémie COVID-19

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


# nastavenie okna aplikácie
st.set_page_config(layout="wide")

# nastavenie tlačidiel
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

# max širka aplikácie
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

model = joblib.load("kmeans_model_4vlna15_.pkl")                # k-means model 4. vlny pre 15 atribútov
scaler = joblib.load("scaler_4vlna15.pkl")                      # štandardizácia atribútov 4. vlny pre 15 atribútov
X_scaled = pd.read_pickle("X_scaled_4vlna15.pkl")               # štandardizované hodnoty 4. vlny pre 15 atribútov
X_orig = pd.read_pickle("X_orig_4vlna15.pkl")                   # originálne, už aj doplnené, hodnoty 4. vlny pre 15 atribútov

# to isté, ale iba pre 14 atributov - bez atribútutu "P-laktát"
model_14 = joblib.load("kmeans_model_4vlna14_.pkl")
scaler_14 = joblib.load("scaler_4vlna14_.pkl")
X_scaled_14 = pd.read_pickle("X_scaled_4vlna14.pkl")
X_orig_14 = pd.read_pickle("X_orig_4vlna14.pkl")

# centroidy modelu
centroids = model.cluster_centers_

# potrebné nastaviť závažnosti zhlukov
cluster_severity = {
    0: "Najmenej závažný",
    1: "Stredne závažný",
    2: "Najzávažnejší"
}

st.title("Fenotypy a kontrafaktuálne vysvetlenia pacientov")

# spočítanie mediánov 
cluster_medians = (
    X_orig
    .assign(cluster=model.labels_)
    .groupby("cluster")
    .median()
)

# Tabuľka medianov fenotypov 
def plot_cluster_heatmap(cluster_medians):
    fig, ax = plt.subplots(figsize=(5, 5.55))

    sns.heatmap(
        cluster_medians.T,      # mediány
        cmap="coolwarm",
        norm=LogNorm(),         # log škála, kôli rôznemu rozsahu hodnôt
        annot=True,             # zobrazí hodnoty
        ax=ax,
        fmt=".2f",              # zaokrúhlenie
        linewidths=1,
        linecolor='gray'
    )

    # vizualizácia
    ax.set_title("Mediány atribútov podľa fenotypov")
    ax.set_ylabel("Atribút")
    ax.set_xlabel("Fenotyp")

    return fig

# Radarový graf štandardizovaných hodnôt pacienta a jeho fenotypu
def plot_radar_scaled(centroids, patient_scaled_df, cluster):

    feature_names = patient_scaled_df.columns.tolist()  # dynamicky sa mení v závislosti zadaných atribútov pri pacientovi
    categories = feature_names                          # názvy atribútov
    N = len(categories)                                 # počet atribútov

    model_used = st.session_state.model_used            # model a centroidy vyberamé podľa použitého modelu vzhľadom na počet vstupných parametrov
    centroids = model_used.cluster_centers_

    cluster_vals = centroids[cluster]                   # hodnoty centroidov
    patient_vals = patient_scaled_df.values.flatten()   # hodnoty pacienta

    # kontrola či sedi pošet atribútov
    if len(cluster_vals) != N:
        st.error("Nesúlad počtu atribútov medzi modelom a pacientom")
        return None

    cluster_vals = cluster_vals.tolist()
    patient_vals = patient_vals.tolist()

    cluster_vals += cluster_vals[:1]
    patient_vals += patient_vals[:1]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]           # uhly grafu podľa počtu atribútov
    angles += angles[:1]

    # tvorba a vizualizácia grafu
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(111, polar=True)

    ax.set_ylim(-3, 3)                                              # rozsah SD v grafe
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(axis='y', labelsize=7, colors="gray")

    ax.plot(angles, cluster_vals, linewidth=1.5, label=(f"Fenotyp {cluster}"))          # vykreslenie centroidu
    ax.fill(angles, cluster_vals, alpha=0.2)

    ax.plot(angles, patient_vals, linewidth=1.5, linestyle="dashed", label="Pacient")   # vykreslenie pacienta

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)                                          # naázvy atribútov okolo kruhu
    ax.set_title("Porovnanie pacienta a jeho fenotypu", pad=30)
    ax.legend(loc="lower right", fontsize = 9, bbox_to_anchor=(1.2, -0.1))              # legenda
    ax.spines["polar"].set_visible(False)
    ax.grid(alpha=0.7)
    
    return fig

# ==========================
# INPUT PACIENTA
# ==========================

# úvodný popis aplikácie
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

# vysvetľujúci info box
info_box("Pre priradenie nového pacienta do existujúceho fenotypu zadajte hodnoty atribútov pacienta. Hodnoty sú predvyplnené mediánmi hodnôt atribútov. Po priradení bude pacient zaradený do jedného z troch zhlukov, teda fenotypov, od najmenej závažného po najzávažnejší fenotyp.")

st.markdown(
        "<p style='font-size:20px; font-weight:600;'>Zadajte údaje pacienta:",
        unsafe_allow_html=True
    )

feature_names = X_orig.columns.tolist()                 # názvy atribútov

feature_order = [                                       # poradie atribútov - na základe požiadaviek lekárov
    "Vek",
    "SatO2 %",
    "WBC last",
    "NE/LY(NLR) last",
    "Fib last",
    "D-dimér HS last",
    "S-Gluk last",
    "S-Urea last",
    "S-Alb last",
    "S-CRP last",
    "S-Na last",
    "S-CL last",
    "S-IL6 last",
    "S-PBNP last",
    "P-Laktát last",
]

feature_units = {                                       # doplnenie jednotiek atribútov
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

feature_ranges = {                                      # povolené rozsahy atribútov - získané od lekárov
    "NE/LY(NLR) last": {"min": 0, "max": 90},
    "S-CRP last": {"min": 0, "max": 2000},
    "S-IL6 last": {"min": 0, "max": 10000},
    "S-Alb last": {"min": 10, "max": 100},
    "S-Na last": {"min": 100, "max": 200},
    "S-Urea last": {"min": 0, "max": 60},
    "D-dimér HS last": {"min": 0, "max": 10},
    "S-PBNP last": {"min": 0, "max": 50000},
    "S-CL last": {"min": 70, "max": 150},
    "SatO2 %": {"min": 30, "max": 100},
    "P-Laktát last": {"min": 0, "max": 15},
    "Fib last": {"min": 0, "max": 15},
    "S-Gluk last": {"min": 0, "max": 50},
    "WBC last": {"min": 0, "max": 60},
    "Vek": {"min": 0, "max": 130},
}

input_data = {}                                         # slovník atribútov a ich vstupných hodnôt

n_cols = 5                                              # polia na zadanie hodnôt pacienta budú v 5 stĺpcoch
cols = st.columns(n_cols)

# prechádzanie vstupných atribútov
for i, feature in enumerate(feature_order):
    col = cols[i % n_cols]                              # rozdelenie atribútov do 5 stĺpcov
    with col:
        unit = feature_units.get(feature, "")           # jednotky atribútu
      
        # ŠPECIÁLNY PRÍPAD: LAKTÁT
        if feature == "P-Laktát last":
            with col:
                lactate_value = st.number_input(                    # input užívateľa, predvyplnená hodnota je medián
                    f"{feature} ({unit})" if unit else feature,
                    value=float(X_orig[feature].median())
                )

            with col:
                lactate_missing = st.checkbox("P-Laktát nevyšetrený", key="lactate_missing")  # checkbox, či je laktát nevyšetrený

            if lactate_missing:
                input_data[feature] = np.nan                        # ak nevyštrený -> nan
            else: 
                input_data[feature] = lactate_value                 # ak vyštrený, berieme vstup užívateľa

        # ostatné atributy
        else:
            # načítanie povolených rozsahov atribútu
            limits = feature_ranges.get(feature, {})
            min_val = float(limits.get("min", None))
            max_val = float(limits.get("max", None))

            input_data[feature] = st.number_input(
                f"{feature} ({unit})" if unit else feature,
                value=float(X_orig[feature].median()),              # input uživateľa, predvyplnená hodnota je medián
                min_value=float(min_val),
                max_value=float(max_val),
                key=f"input_{feature}"                              
            )

new_patient_df = pd.DataFrame([input_data])                         # convert na DataFrame


# INICIALIZACIA zhluku, škálovaných dát pacienta a originálnych dát pacienta

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
    pressed = st.button("Priradiť pacienta", use_container_width=True)      # tlačidlo Priradiť pacienta

    if pressed:
        new_patient_df = pd.DataFrame([input_data])                         # nahranie dát zo vstupu

        # Prepínanie modelov vzhľadom na vyšetrenie atribútu P-Laktát
        if pd.isna(new_patient_df["P-Laktát last"].iloc[0]):                # pacient nemá P-Laktát  

            df = new_patient_df.drop(columns=["P-Laktát last"])             # odstránime aj atribút P-Laktát
            df = df[X_scaled_14.columns]                                    # zoradenie stĺpcov aby sa zhodovalo s tými v modeli

            new_patient_scaled = scaler_14.transform(df)                    # škálovanie hodnôt pacienta podľa scaleru

            new_patient_scaled_df = pd.DataFrame(                           # škálované hodnoty pacienta v df
                new_patient_scaled,
                columns=X_scaled_14.columns
            )

            cluster = model_14.predict(new_patient_scaled_df)[0]            # predikcia zhluku podľa modelu, výsledok je číslo zhluku

            # uložeenie správneho modelu a hodnôt
            st.session_state.X_scaled_used = X_scaled_14
            st.session_state.model_used = model_14
            st.session_state.X_orig_used = X_orig_14

        else:                                                               # pacient má P-Laktát

            df = new_patient_df[X_scaled.columns]                           # zoradenie stĺpcov
            new_patient_scaled = scaler.transform(df)                       # škálovanie hodnôt pacienta podľa scaleru

            new_patient_scaled_df = pd.DataFrame(                           # škálované hodnoty pacienta v df
                new_patient_scaled,
                columns=X_scaled.columns
            )

            cluster = model.predict(new_patient_scaled_df)[0]               # predikcia zhluku podľa modelu, výsledok je číslo zhluku

            # uložeenie správneho modelu a hodnôt
            st.session_state.X_scaled_used = X_scaled
            st.session_state.model_used = model
            st.session_state.X_orig_used = X_orig

        # uloženie zhluku, škálovaných dát pacienta a originálnych dát pacienta
        st.session_state.cluster = cluster
        st.session_state.new_patient_scaled_df = new_patient_scaled_df
        st.session_state.new_patient_df = new_patient_df



# ZOBRAZENIE VÝSLEDKU Radaroveho grafu a tabulky medianov
if st.session_state.cluster is not None:

    cluster = st.session_state.cluster                                      # číslo zhluku

    st.markdown(                                                            # popis
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

    # vysvetľujúci info box
    info_box(f"Graf naľavo zobrazuje porovnanie štandardizovaných hodnôt pacienta a centroidu jeho fenotypu, teda fenotypu {cluster}. Tabuľka napravo zobrazuje mediány hodnôt atribútov jednotlivých fenotypov.")

    left_panel, right_panel = st.columns([1,1])                             # rodelenie obrazovky na 2 časti

    with left_panel:
        radar_fig = plot_radar_scaled(                                      # radarový graf naľavo
            centroids,
            st.session_state.new_patient_scaled_df,
            cluster
        )
        st.pyplot(radar_fig, use_container_width=False)

    with right_panel:
        heatmap_fig = plot_cluster_heatmap(cluster_medians)                 # tabuľka mediánov atribútov napravo
        st.pyplot(heatmap_fig, use_container_width=False)

    possible_targets = [c for c in cluster_severity.keys() if c != cluster] # možné cieľové zhluky sú všetky okrem aktuálneho

    st.divider()

    st.subheader("2. časť - Kontrafaktuálne vysvetlenia")

    # vysvetľujúci info box
    info_box(f"Kontrafaktuálne vysvetlenia popisujú aké minimálne zmeny by museli u daného pacienta nastať aby patril do iného fenotypu. Zo zoznamu vyberte fenotyp, pre ktorý chcete tieto minimálne zmeny určiť. Výsledkom budú minimálne zmeny vzhľadom ku najbližšiemu pacientovi (kontrafaktuálny pacient) vo vybranom fenotype.")

    st.markdown(
        "<p style='font-size:20px; font-weight:600;margin-bottom:-40px;'>Vyberte cieľový fenotyp:",
        unsafe_allow_html=True
    )

    desired_cluster = st.selectbox(                                         # používateľ si môže vybrať do ktorého zhluku presunie pacienta 
        "",
        possible_targets,
        format_func=lambda x: f"{x} – {cluster_severity[x]}"
    )

    
    # KONTRAFAKTUÁL - Výpočet a zobrazenie
    
    cf_left, cf_center, cf_right = st.columns([2,3,2])                              # rozdelenie obrazovky na 3 časti

    with cf_center:
        pressed2 = st.button("Vypočítať kontrafaktuálne vysvetlenia", use_container_width=True)     # tlačidlo
        
        if pressed2:

            new_patient_scaled_df = st.session_state.new_patient_scaled_df          # škálované hodnoty pacienta
            new_patient_df = st.session_state.new_patient_df                        # hodnoty pacienta
            model_used = st.session_state.model_used                                # požitý model
            X_scaled_used = st.session_state.X_scaled_used                          # štandardizované hodnoty pacientov modeli

            centroids = model_used.cluster_centers_                                 # centroidy

            mask = model_used.labels_ == desired_cluster
            X_target_scaled = X_scaled_used[mask]                                   # štandardizované hodnoty pacientov s cieľového zhluku

            nn = NearestNeighbors(n_neighbors=1)                                    # trénovanie modelu na cieľovom zhluku
            nn.fit(X_target_scaled.values)

            dist, neighbor_pos = nn.kneighbors(new_patient_scaled_df.values)        # najbližší pacient v cieľovom zhluku
            anchor_idx = X_target_scaled.index[neighbor_pos[0][0]]                  # index toho pacienta

            x_anchor_scaled = X_scaled_used.loc[anchor_idx].values                  # pacient v cieľovom zhluku
            x_patient_scaled = new_patient_scaled_df.values.ravel()                 # zadaný pacient

            deltas = x_anchor_scaled - x_patient_scaled                             # rozdiely v ich atribútoch
            order = np.argsort(-np.abs(deltas))                                     # od najväčšieho rozdielu

            x_cf_scaled = x_patient_scaled.copy()
            changed = []

            feature_names_used = new_patient_scaled_df.columns.tolist()             # zoznam použitých atribútov

            # prechádzame zmenami medzi atribútmi
            for j in order:
                x_cf_scaled[j] = x_anchor_scaled[j]                                 # zmena atribútu podľa pacienta v cieľovom zhluku
                changed.append(feature_names_used[j])                               # zaznamenaie zmien

                new_cluster = np.argmin(
                    np.linalg.norm(centroids - x_cf_scaled, axis=1)                 # zisťujeme ku ktorému zhluku po zmene patrí
                )

                if new_cluster == desired_cluster:                                  # ak už patrí do cieľového zhluku, tak koniec
                    break
            
            X_orig_used = st.session_state.X_orig_used                              # hodnoty pacientov modeli

            x_anchor_orig = X_orig_used.loc[anchor_idx]                             # hodnoty najbližšieho pacienta z cieľového zhluku
            x_cf_orig = new_patient_df.iloc[0].copy()                               # hodnoty zadaného pacienta

            for col in changed:
                x_cf_orig[col] = x_anchor_orig[col]                                 # iba hodnoty atribútov pacienta z ceľového zhluku, ktoré boli v zadanom pacientovi zmenené

            delta_orig = x_cf_orig - new_patient_df.iloc[0]                         # rozdiely v atribútoch pacientov v reálnych jednotkách

            cf_table = pd.DataFrame({                                               # tabuľka zadaného pacienta, kontrafaktuálneho a rozdielov ich atribútov
                "Nový pacient": new_patient_df.iloc[0],
                "Kontrafaktuálny pacient": x_cf_orig,
                "Zmena": delta_orig
            })

            cf_table = cf_table.loc[changed].sort_values(                           # iba rozdiely zmenených atribútov zoradené podľa veľkosti
                by="Zmena",
                key=lambda s: s.abs(),
                ascending=False
            )

            with cf_center:                                                         # zobrazenie v strednej časti obrazovky
                st.markdown("##### Minimálne zmeny pre preradenie pacienta:")
                st.dataframe(cf_table, use_container_width=True)
