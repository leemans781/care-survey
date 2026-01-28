# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 13:12:48 2026

@author: SashaLeemans
"""

# app.py — Streamlit AHP survey met integratie van de code van care 0.005 
import os
import streamlit as st
import numpy as np
import pandas as pd
from numpy.linalg import eigvals


st.set_page_config(page_title="AHP Survey", layout="wide")

RESP_DIR = "responses"  # map waarin CSV's van deelnemers komen
os.makedirs(RESP_DIR, exist_ok=True)

ADMIN_CODE = "secret123"

# Dit moet ervoor zorgen dat de respondenten pas kunnen invullen, als de beheerder de criteria heeft ingevuld
# Er valt namelijk nog niks in te vullen, als de criteria nog niet bekend zijn
# Dit zelfde pas ik doe voor de alternatieven. 
if "criteria" not in st.session_state:
    st.session_state.criteria = []

if "criteria_locked" not in st.session_state:
    st.session_state.criteria_locked = False
       
if "alternatives" not in st.session_state:
    st.session_state.alternatives = []
    
if "alternatives_locked" not in st.session_state:
    st.session_state.alternatives_locked = False
    


def weights_colmean(M: np.ndarray) -> np.ndarray:
    """Bereken AHP-gewichten via kolomnormalisatie + rijgemiddelden (som=1)."""
    colsum = M.sum(axis=0)
    norm = M / colsum
    w = norm.mean(axis=1)
    return w / w.sum()

def saaty_cr(M: np.ndarray, w: np.ndarray) -> float:
    """Saaty Consistency Ratio (voor n<=10)."""
    lam = np.mean((M @ w) / w)
    n = M.shape[0]
    ri_map = {1:0.0, 2:0.0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    ri = ri_map.get(n, None)
    if ri is None:
        # Voor n>10: advies is Alonso-Lamata
        return np.nan
    ci = (lam - n) / (n - 1)
    return 0.0 if ri == 0 else ci / ri

def alo_cr(M: np.ndarray) -> float:
    """Alonso-Lamata CR (voor n>10)."""
    n = M.shape[0]
    lam = max(np.real(eigvals(M)))
    return (lam - n) / (2.7699 * n - 4.3513 - n)

def consolidate_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    """AIJ: element-wise product + k-de wortel (geometrisch gemiddelde) tot consolidated matrix."""
    if not matrices:
        return None
    P = np.ones_like(matrices[0], dtype=float)
    for A in matrices:
        P *= A
    k = len(matrices)
    G = P ** (1.0 / k)
    return G


def calculate_homogeneity(priorities: list[dict]) -> float:
    """
    Bereken homogeniteit op basis van Shannon-entropie.
    priorities = lijst van dicts met gewichten per respondent.
    """
    n = len(priorities)  # aantal respondenten
    cats = list(priorities[0].keys())  # categorieën
    catCnt = len(cats)

    # Gemiddelde distributie (groep)
    avg = {c: 0.0 for c in cats}
    for p in priorities:
        for c in cats:
            avg[c] += p[c]
    for c in cats:
        avg[c] /= n

    # Entropie van groep (gamma)
    gamma = -sum(avg[c] * np.log(avg[c]) for c in cats if avg[c] > 0)

    # Entropie van individuen (alpha)
    alpha_sum = 0.0
    for p in priorities:
        alpha_sum += -sum(p[c] * np.log(p[c]) for c in cats if p[c] > 0)
    alpha = alpha_sum / n

    # Beta = verschil
    beta = gamma - alpha
    sim = (1. / np.exp(beta) - 1. / catCnt) / (1. - 1. / catCnt)  # schaal zoals in jouw collega’s code
    return sim

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Bereken cosine similarity matrix zonder sklearn.
    X: shape (n_respondenten, n_criteria)
    """
    # Normaliseer elke vector
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / norms

    # Cosine similarity = dot product van genormaliseerde vectors
    return X_norm @ X_norm.T


def calculate_consensus(priorities: list[dict]) -> float:
    """
    Bereken consensus via gemiddelde cosine similarity (zonder sklearn).
    """
    cats = list(priorities[0].keys())
    vectors = np.array([[p[c] for c in cats] for p in priorities])

    cosine_sim_matrix = cosine_similarity_matrix(vectors)

    # Gemiddelde off-diagonal similarity
    total = 0.0
    count = 0
    for i in range(cosine_sim_matrix.shape[0]):
        for j in range(cosine_sim_matrix.shape[1]):
            if i != j:
                total += cosine_sim_matrix[i, j]
                count += 1

    return total / count if count > 0 else 0.0


def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

# -----------------------------
# Sidebar: moduskeuze
# -----------------------------
st.sidebar.header("Navigatie")
mode = st.sidebar.radio("Kies modus", ["Deelnemer (invullen)", "Admin (groepresultaat)"])

# -----------------------------
# Criteria invoer (geldt voor beide modi)
# -----------------------------
# st.header("AHP Pairwise Survey")

criteria = st.session_state.criteria
n = len(criteria)

st.header("AHP Pairwise Survey")
st.write("### Criteria")
st.write(", ".join(criteria))


# -----------------------------
# DEELNEMER MODUS
# -----------------------------
if mode == "Deelnemer (invullen)":
     
    if not (st.session_state.criteria_locked and st.session_state.alternatives_locked):
        st.warning("De survey is nog niet geopend.")
        st.stop()

    tabs = st.tabs(["Criteria", "Alternatieven"])
    with tabs[0]:
        st.subheader("Deelnemer — pairwise vergelijkingen invullen")
    
        participant_name = st.text_input("Jouw naam (of e-mail)", placeholder="Naam of e-mail")
        if not participant_name:
            st.info("Vul je naam/e-mail in om verder te gaan.")
            st.stop()
    
        st.write(f"Aantal criteria: **{n}**")
        st.markdown("---")
    
        # Boven-driehoek invoer
        vals = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{criteria[i]} vs {criteria[j]}"
                with st.container():
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        side = st.radio(
                            key,
                            [
                                f"{criteria[i]} > {criteria[j]}",
                                f"{criteria[i]} ≈ {criteria[j]}",
                                f"{criteria[j]} > {criteria[i]}",
                            ],
                            index=0,
                            horizontal=True,
                        )
                    with col2:
                        mag = st.slider("Sterkte (1 = zwak, 9 = zeer sterk)", 1, 9, 3, key=key + "_mag")
    
                    # if "≈" in side:
                    #     vals[(i, j)] = 1.0
                    # elif criteria[i] in side:  # i > j
                    #     vals[(i, j)] = float(mag)
                    # else:  # j > i
                    #     vals[(i, j)] = 1.0 / float(mag)
                        
                    # 🔧 HIER zit oplossing 1
                    if "≈" in side:
                        v = 1.0
                    elif side.startswith(criteria[i]):  # i > j
                        v = float(mag)
                    else:  # j > i
                        v = 1.0 / float(mag)
        
                    vals[(i, j)] = v
    
        # Volledige matrix opbouwen
        A = np.ones((n, n), dtype=float)
        for (i, j), v in vals.items():
            A[i, j] = v
            A[j, i] = 1.0 / v
    
        st.write("**Jouw pairwise matrix**")
        st.dataframe(pd.DataFrame(A, columns=criteria, index=criteria))
    
        # Gewichten + CR
        w = weights_colmean(A)
        cr = saaty_cr(A, w) if n <= 10 else alo_cr(A)
    
        st.subheader("Jouw prioriteiten")
        st.write(pd.DataFrame({"Criteria": criteria, "Weight (%)": (w * 100).round(2)}))
        st.metric("Consistency Ratio (CR)", f"{cr * 100:.1f}%")
    
        # Opslaan
        st.markdown("---")
        if st.button("Verstuur en opslaan"):
            # Bestandsnaam veilig maken
            safe_name = "".join(ch for ch in participant_name if ch.isalnum() or ch in ("_", "-", "."))
            out_path = os.path.join(RESP_DIR, f"{safe_name}.csv")
            pd.DataFrame(A, columns=criteria, index=criteria).to_csv(out_path, index=True)
            st.success(f"Inzending opgeslagen: `{out_path}`")
            st.info("Je kunt het tabblad sluiten. Bedankt voor het invullen!")
        
    with tabs[1]:
        if not st.session_state.criteria_locked:
            st.warning("Alternatieven kunnen pas ingevuld worden als de criteria zijn afgerond.")
        else:
            st.subheader("Alternatieven — pairwise vergelijkingen per criterium")
    
            participant_name = st.text_input("Jouw naam/e-mail (voor alternatieven)", placeholder="Naam of e-mail", key="alt_name")
            if not participant_name:
                st.info("Vul je naam/e-mail in om verder te gaan.")
                st.stop()
    
            alternatives = st.session_state.alternatives
            criteria = st.session_state.criteria
    
            # Loop over elk criterium
            for crit in criteria:
                st.markdown(f"### Alternatieven voor criterium: **{crit}**")
                
                n_alt = len(alternatives)
                vals = {}
    
                # Boven-diagonaal invoer van alternatieven
                for i in range(n_alt):
                    for j in range(i+1, n_alt):
                        key = f"{crit}_{alternatives[i]}_vs_{alternatives[j]}"
                        with st.container():
                            col1, col2 = st.columns([3,2])
                            with col1:
                                side = st.radio(
                                    key,
                                    [
                                        f"{alternatives[i]} > {alternatives[j]}",
                                        f"{alternatives[i]} ≈ {alternatives[j]}",
                                        f"{alternatives[j]} > {alternatives[i]}"
                                    ],
                                    index=0,
                                    horizontal=True
                                )
                            with col2:
                                mag = st.slider("Sterkte (1 = zwak, 9 = zeer sterk)", 1, 9, 3, key=key + "_mag")
                        
                        # Bereken waarde
                        if "≈" in side:
                            v = 1.0
                        elif side.startswith(alternatives[i]):
                            v = float(mag)
                        else:
                            v = 1.0 / float(mag)
                        
                        vals[(i,j)] = v
    
                # Bouw volledige matrix
                A = np.ones((n_alt, n_alt), dtype=float)
                for (i,j), v in vals.items():
                    A[i,j] = v
                    A[j,i] = 1.0 / v
    
                st.write("**Jouw pairwise matrix voor dit criterium**")
                st.dataframe(pd.DataFrame(A, columns=alternatives, index=alternatives))
    
                # Gewichten + CR
                w = weights_colmean(A)
                cr = saaty_cr(A, w) if n_alt <= 10 else alo_cr(A)
    
                st.subheader("Prioriteiten voor dit criterium")
                st.write(pd.DataFrame({"Alternatief": alternatives, "Weight (%)": (w*100).round(2)}))
                st.metric("Consistency Ratio (CR)", f"{cr*100:.1f}%")
    
            # Opslaan van alle alternatieven-matrices per criterium
            st.markdown("---")
            if st.button("Verstuur en opslaan alternatieven"):
                safe_name = "".join(ch for ch in participant_name if ch.isalnum() or ch in ("_", "-", "."))
                alt_dir = os.path.join(RESP_DIR, "alternatives")
                os.makedirs(alt_dir, exist_ok=True)
    
                for crit in criteria:
                    # Hier zou je dezelfde matrices A moeten opslaan per criterium
                    out_path = os.path.join(alt_dir, f"{safe_name}_{crit}.csv")
                    pd.DataFrame(A, columns=alternatives, index=alternatives).to_csv(out_path, index=True)
                st.success(f"Inzendingen voor alle criteria opgeslagen in '{alt_dir}'")
                st.info("Bedankt voor het invullen van de alternatieven!")


    
# -----------------------------
# ADMIN MODUS
# -----------------------------
else:
    st.subheader("Admin — groepsresultaat (alleen voor jou)")

    code = st.text_input("Admin code", type="password")
    if code != ADMIN_CODE:
        st.warning("Voer de juiste admin code in om groepsresultaat te zien.")
        st.stop()
        
    st.markdown("### Criteria instellen (alleen admin)")
    criteria_input = st.text_area(
        "Voer criteria in (één per regel)",
        value="\n".join(st.session_state.criteria)
    )
    
    if st.button("Criteria opslaan"):
        new_criteria = [c.strip() for c in criteria_input.splitlines() if c.strip()]
        if len(new_criteria) < 2:
            st.error("Minimaal 2 criteria vereist.")
        else:
            st.session_state.criteria = new_criteria
            #st.session_state.criteria_locked = True
            st.success("Criteria opgeslagen.")    
            
    # De volgende stap: alternatieven introduceren. 
    st.markdown("### Alternatieven instellen (alleen admin)")

    alternatives_input = st.text_area("Voer alternatieven in (één per regel)", value="\n".join(st.session_state.alternatives))
    
    if st.button("Alternatieven opslaan"):
        new_alternatives = [a.strip() for a in alternatives_input.splitlines() if a.strip()]
        
        if len(new_alternatives) < 2:
            st.error("Minimaal 2 alternatieven vereist.")
        else:
            st.session_state.alternatives = new_alternatives
            st.success("Alternatieven opgeslagen.")
            
    if (
        len(st.session_state.criteria) >= 2
        and len(st.session_state.alternatives) >= 2
    ):
        st.session_state.criteria_locked = True
        st.session_state.alternatives_locked = True


    # Lees alle responses die overeenkomen met deze criteria-dimensie (n x n)
    files = [f for f in os.listdir(RESP_DIR) if f.endswith(".csv")]
    st.write(f"Gevonden inzendingen: **{len(files)}**")
    matrices = []
    bad_files = []
    for f in files:
        path = os.path.join(RESP_DIR, f)
        try:
            df = pd.read_csv(path, index_col=0)
            # controle: zelfde dimensie en zelfde criteria set?
            if df.shape != (n, n):
                bad_files.append(f"{f} (dimensie {df.shape}, verwacht {(n,n)})")
                continue
            # optioneel: check index/kolomnamen gelijk aan criteria
            # Bij afwijkende namen kun je alleen op dimensie matchen in MVP
            A = df.values.astype(float)
            matrices.append(A)
        except Exception as e:
            bad_files.append(f"{f} (error: {e})")

    if bad_files:
        st.warning("Sommige files konden niet gebruikt worden:\n- " + "\n- ".join(bad_files))

    if not matrices:
        st.info("Geen bruikbare responses gevonden in de map 'responses/'.")
        st.stop()

    # Consolidated decision matrix via AIJ
    G = consolidate_matrices(matrices)
    st.write("**Consolidated Decision Matrix (groep)**")
    st.dataframe(pd.DataFrame(G, columns=criteria, index=criteria))

    # Groepsgewichten + CR
    wg = weights_colmean(G)
    group_cr = saaty_cr(G, wg) if n <= 10 else alo_cr(G)

    # Overzichtstabel
    st.subheader("Groepsprioriteiten")
    df_grp = pd.DataFrame({
        "Criteria": criteria,
        "Weight (%)": (wg * 100).round(2)
    })
    # Rang bepalen
    df_grp["Rank"] = df_grp["Weight (%)"].rank(ascending=False, method="dense").astype(int)
    st.write(df_grp)

    #st.metric("Group Consistency Ratio (CR)", f"{group_cr * 100:.1f}%")
    
    # Bereken homogeniteit en consensus
    priorities_list = []
    for f in files:
        df = pd.read_csv(os.path.join(RESP_DIR, f), index_col=0)
        weights = weights_colmean(df.values)
        priorities_list.append({criteria[i]: weights[i] for i in range(len(criteria))})
    
    homogeneity = calculate_homogeneity(priorities_list)
    consensus = calculate_consensus(priorities_list)
    
    st.subheader("Groepshomogeniteit & Consensus")
    st.metric("Homogeniteit (S)", f"{homogeneity:.3f}")
    st.metric("Consensus (S*)", f"{consensus:.3f}")
    
    # Interpretatie
    def interpret(value):
        if value >= 0.8:
            return "Hoog"
        elif value >= 0.6:
            return "Matig"
        else:
            return "Laag"
    
    st.write(f"Interpretatie homogeniteit: {interpret(homogeneity)}")
    st.progress(homogeneity)
    st.write(f"Interpretatie consensus: {interpret(consensus)}")
    st.progress(consensus)

    # Export knoppen
    st.markdown("---")
    colA, colB, colC = st.columns(3)
    with colA:
        # consolidated matrix download
        cm_bytes = pd.DataFrame(G, columns=criteria, index=criteria).to_csv(index=True).encode("utf-8")
        st.download_button("Download consolidated matrix (CSV)", cm_bytes, file_name="consolidated_matrix.csv")
    with colB:
        # group weights download
        gw_bytes = df_grp.to_csv(index=False).encode("utf-8")
        st.download_button("Download group weights (CSV)", gw_bytes, file_name="group_weights.csv")
    with colC:
        # log van gebruikte files
        log_bytes = "\n".join(files).encode("utf-8")
        st.download_button("Download list of participant files (TXT)", log_bytes, file_name="participants_used.txt")

    st.caption("MVP: responses staan lokaal in de map 'responses/'. In Streamlit Cloud blijven ze bewaard zolang de app niet opnieuw wordt gedeployed. Voor productie: gebruik een database of Blob Storage.")


    
