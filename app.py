"""Clinician-facing demo.  streamlit run app.py -- --ckpt runs/exp1/best.pt"""
import argparse
import sys
import tempfile

import streamlit as st

from medscan.pipeline import Analyzer

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/default/best.pt")
args, _ = p.parse_known_args(sys.argv[1:])

st.set_page_config(page_title="MedScan decision support", layout="wide")
st.title("Adaptive AI medical scan analysis — decision support")
st.caption("Research prototype. Not a medical device; the clinician makes the final decision.")


@st.cache_resource
def get_analyzer(path):
    return Analyzer(path)


an = get_analyzer(args.ckpt)
up = st.file_uploader("Upload scan", type=["png", "jpg", "jpeg", "dcm", "nii", "gz"])
c1, c2 = st.columns(2)
task = c1.selectbox("Task", ["(auto-detect modality)"] + list(an.tasks))
pid = c2.text_input("Patient ID (optional, stored hashed; enables comparison with prior study)")

if up and st.button("Analyse"):
    suffix = "." + up.name.split(".", 1)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(up.read())
    res, ov, _ = an.analyze(f.name, None if task.startswith("(") else task, pid or None)
    st.session_state["res"], st.session_state["ov"] = res, ov

if "res" in st.session_state:
    res, ov = st.session_state["res"], st.session_state["ov"]
    a, b = st.columns(2)
    a.image(ov, caption=f"Grad-CAM for '{res['gradcam_target']}'", use_container_width=True)
    color = {"high": "green", "medium": "orange", "low": "red"}[res["tier"]]
    b.markdown(f"**Modality:** {res['modality']}  \n**Task:** {res['task']}  \n"
               f"**Prediction:** {res['prediction']}  \n**Confidence:** {res['confidence']:.1%} "
               f"(:{color}[{res['tier']}])")
    b.info(res["action"])
    for n in res["notes"]:
        b.warning(n)
    q = res["quality"]
    b.write(f"Scan quality score {q['score']:.2f} — blur {q['blur']:.0f}, contrast {q['contrast']:.0f}, "
            f"edge density {q['edge_density']:.2f}")
    b.bar_chart(res["probabilities"])
    if res["longitudinal"]:
        b.write("Change vs previous study (probability delta):", res["longitudinal"]["prob_change"])
    st.subheader("Clinician feedback")
    agree = st.radio("Do you agree with the AI suggestion?", ["Yes", "No"], horizontal=True)
    final = st.text_input("Final diagnosis (if different)")
    note = st.text_area("Comment")
    if st.button("Submit feedback"):
        an.audit.feedback(res["case_id"], agree == "Yes", final or None, note)
        st.success("Feedback recorded in the audit log.")
