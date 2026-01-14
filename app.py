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

# Dit zijn de default criteria die worden gebruikt, als er niks is doorgevoerd
# Ik vind dit eigenlijk een lelijke manier van programmeren, maar weet even niks anders
if "criteria" not in st.session_state:
    st.session_state.criteria = ["Kosten", "Bereikbaarheid", "Duurzaamheid"]

RESP_DIR = "responses"  # map waarin CSV's van deelnemers komen
os.makedirs(RESP_DIR, exist_ok=True)

ADMIN_CODE = "secret123"

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
st.header("AHP Pairwise Survey")
criteria = st.session_state.criteria
n = len(criteria)

st.write("### Criteria")
st.write(", ".join(criteria))


# -----------------------------
# DEELNEMER MODUS
# -----------------------------
if mode == "Deelnemer (invullen)":
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

                if "≈" in side:
                    vals[(i, j)] = 1.0
                elif criteria[i] in side:  # i > j
                    vals[(i, j)] = float(mag)
                else:  # j > i
                    vals[(i, j)] = 1.0 / float(mag)

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
            st.success("Criteria succesvol opgeslagen!")    

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

    st.metric("Group Consistency Ratio (CR)", f"{group_cr * 100:.1f}%")

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
