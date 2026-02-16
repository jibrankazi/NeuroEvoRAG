import os
import pickle

import pandas as pd
import streamlit as st

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "experiment_results.pkl",
)


def load_results(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def render_no_results():
    st.warning(
        "No experiment results found. Run an evolution experiment first "
        "to generate outputs/experiment_results.pkl."
    )
    st.stop()


def render_config(config: dict):
    st.subheader("Experiment Configuration")
    col1, col2, col3 = st.columns(3)
    col1.metric("Generations", config.get("num_generations", "N/A"))
    col2.metric("Population Size", config.get("population_size", "N/A"))
    col3.metric("Evaluation Samples", config.get("num_samples", "N/A"))


def render_fitness_chart(history: list[dict]):
    st.subheader("Fitness Over Generations")
    df = pd.DataFrame(history)
    best_col = "best_fitness" if "best_fitness" in df.columns else "best_score"
    avg_col = "avg_fitness" if "avg_fitness" in df.columns else "avg_score"
    chart_df = df[["generation", best_col, avg_col]].copy()
    chart_df = chart_df.rename(
        columns={best_col: "Best Fitness", avg_col: "Average Fitness"}
    )
    chart_df = chart_df.set_index("generation")
    st.line_chart(chart_df)


def render_history_table(history: list[dict]):
    st.subheader("Evolution History")
    rows = []
    for entry in history:
        genome = entry.get("best_genome", {})
        best_key = "best_fitness" if "best_fitness" in entry else "best_score"
        avg_key = "avg_fitness" if "avg_fitness" in entry else "avg_score"
        row = {
            "Generation": entry["generation"],
            "Best Fitness": round(entry[best_key], 4),
            "Avg Fitness": round(entry[avg_key], 4),
        }
        if "best_f1" in entry:
            row["Best F1"] = round(entry["best_f1"], 4)
        if "best_em" in entry:
            row["Best EM"] = round(entry["best_em"], 4)
        row["Best Genome"] = str(genome)
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_comparison(baseline: dict, evolved: dict):
    st.subheader("Baseline vs Evolved")

    b_fit = baseline.get("fitness", baseline.get("score", 0))
    e_fit = evolved.get("fitness", evolved.get("score", 0))
    b_config = baseline.get("config", baseline)
    e_config = evolved.get("config", evolved.get("genome", {}))
    b_metrics = baseline.get("metrics", {})
    e_metrics = evolved.get("metrics", {})

    metrics_df = pd.DataFrame({
        "Baseline": [b_fit, b_metrics.get("f1", 0), b_metrics.get("exact_match", 0)],
        "Evolved": [e_fit, e_metrics.get("f1", 0), e_metrics.get("exact_match", 0)],
    }, index=["Fitness", "F1", "Exact Match"])
    st.bar_chart(metrics_df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline Configuration**")
        st.json({
            "chunk_size": b_config.get("chunk_size"),
            "top_k": b_config.get("top_k"),
            "temperature": b_config.get("temperature"),
            "fitness": round(b_fit, 4) if isinstance(b_fit, float) else b_fit,
        })
    with col2:
        st.markdown("**Evolved Configuration**")
        st.json({
            "chunk_size": e_config.get("chunk_size"),
            "top_k": e_config.get("top_k"),
            "temperature": e_config.get("temperature"),
            "fitness": round(e_fit, 4) if isinstance(e_fit, float) else e_fit,
        })

    delta = e_fit - b_fit
    if delta > 0:
        st.success(f"Evolution improved fitness by +{delta:.4f}")
    elif delta < 0:
        st.error(f"Evolved fitness decreased by {delta:.4f}")
    else:
        st.info("Evolved fitness is equal to baseline.")


def render_best_genome(evolved: dict):
    st.subheader("Best Genome Found")
    genome = evolved.get("config", evolved.get("genome", {}))
    if not genome:
        st.write("No genome data available.")
        return
    cols = st.columns(len(genome))
    for col, (key, value) in zip(cols, genome.items()):
        col.metric(key, value)


def main():
    st.set_page_config(page_title="NeuroEvoRAG Dashboard", layout="wide")
    st.title("NeuroEvoRAG -- Evolution Dashboard")

    results = load_results(RESULTS_PATH)

    if results is None:
        render_no_results()
        return

    baseline = results.get("baseline", {})
    evolved = results.get("evolved", {})
    history = results.get("history", [])
    config = results.get("experiment", results.get("config", {}))

    if config:
        render_config(config)

    st.divider()

    if history:
        render_fitness_chart(history)
        st.divider()
        render_history_table(history)
        st.divider()

    render_comparison(baseline, evolved)
    st.divider()
    render_best_genome(evolved)


if __name__ == "__main__":
    main()
