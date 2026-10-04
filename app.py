# Standard-library path handling keeps the model directory relative to this app file.
from pathlib import Path

import pandas as pd
import streamlit as st

import portable


# The app reads the assignment's public synthetic student files.
DATA_URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"
# Keep this feature order identical to the columns used to train and export the model.
# Post-week-6 fields, identifiers, and the outcome are intentionally excluded.
FEATURES = [
    "age",
    "admission_grade",
    "international",
    "first_gen",
    "su_scholarship",
    "fees_owed",
    "moved_from_home",
    "married",
    "evening_programme",
    "programme",
    "gender",
    "logins_total",
    "logins_last3",
    "logins_trend",
    "submitted_share",
    "missed_last3",
    "quiz_mean",
    "weeks_since_login",
]

# Configure the page before drawing any Streamlit elements.
st.set_page_config(page_title="Study Office | Week 6", layout="wide")
st.title("Study Office | Week 6")
st.caption(
    "A human-reviewed support-priority list. A risk score is not a decision or a diagnosis."
)


# Cache downloaded tables so changing a widget does not fetch the CSVs again.
@st.cache_data
def load_data():
    history = pd.read_csv(f"{DATA_URL}history_week6.csv")
    current = pd.read_csv(f"{DATA_URL}new_week6.csv")
    return history, current


# Cache the portable model as a resource; it is loaded once per app process.
@st.cache_resource
def load_model():
    return portable.Model(str(Path(__file__).parent / "model"))


# Score a copy so the original source data stays unchanged.
def add_risk(frame, model):
    scored = frame.copy()
    # The portable model returns each student's probability of leaving.
    scored["risk"] = pd.Series(
        model.predict_proba(scored[FEATURES]), index=scored.index, dtype="float64"
    )
    # Rank high risk first; "first" makes tied scores resolve consistently.
    scored["priority"] = scored["risk"].rank(method="first", ascending=False).astype(int)
    return scored


# Load past outcomes for validation and this week's students for the live list.
# Only the 2025 cohort is used to evaluate the final contact rule.
try:
    history, current = load_data()
    model = load_model()
    validation = add_risk(history.loc[history["cohort"] == 2025].copy(), model)
    current = add_risk(current, model)
except Exception as error:
    # Stop here rather than showing an empty or misleading dashboard.
    st.error(f"The data or model could not be loaded: {error}")
    st.stop()

# These are the assignment's editable DKK assumptions; each selected student incurs a conversation cost.
max_contacts = max(40, len(current))
st.sidebar.subheader("Decision-cost assumptions (DKK)")
false_alarm_cost_dkk = st.sidebar.number_input(
    "False alarm cost (DKK)", min_value=0, value=2_000, step=500
)
student_leaving_cost_dkk = st.sidebar.number_input(
    "Cost of a student leaving (DKK)", min_value=0, value=60_000, step=5_000
)
conversation_cost_dkk = st.sidebar.number_input(
    "Cost per conversation (DKK)", min_value=0, value=500, step=100
)
conversation_help_rate = st.sidebar.slider(
    "Share of at-risk students helped",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
    format="percent",
)
# Start at the assignment's 40-contact capacity and keep both widget values in session state.
default_capacity = min(40, max_contacts)
st.session_state.setdefault("capacity_slider", default_capacity)
st.session_state.setdefault("capacity_input", default_capacity)
st.session_state["capacity_slider"] = min(
    max_contacts, max(0, st.session_state["capacity_slider"])
)
st.session_state["capacity_input"] = min(
    max_contacts, max(0, st.session_state["capacity_input"])
)
# Reconcile stale values when Streamlit reloads the script after an app-file change.
st.session_state["capacity_input"] = st.session_state["capacity_slider"]


# Each callback copies the changed widget's value into its sibling before rerunning the app.
def sync_capacity_input():
    st.session_state["capacity_input"] = st.session_state["capacity_slider"]


def sync_capacity_slider():
    st.session_state["capacity_slider"] = st.session_state["capacity_input"]


# Offer both a draggable control and direct integer entry for the same capacity.
st.sidebar.slider(
    "Conversations available",
    min_value=0,
    max_value=max_contacts,
    key="capacity_slider",
    on_change=sync_capacity_input,
    help="The same capacity rule is evaluated on the 2025 validation cohort below.",
)
st.sidebar.number_input(
    "Or enter a conversation count",
    min_value=0,
    max_value=max_contacts,
    step=1,
    key="capacity_input",
    on_change=sync_capacity_slider,
)
contact_capacity = int(st.session_state["capacity_input"])

# Apply the capacity rule to both datasets: choose the highest-ranked students.
current["selected"] = current["priority"] <= contact_capacity
validation["contacted"] = validation["priority"] <= min(contact_capacity, len(validation))

# Evaluate candidate risk cutoffs on 2025. A contact costs DKK 500 whether that student
# stays or leaves; a false alarm adds its DKK 2,000 worry cost. Contacting a leaver has
# an expected benefit based on the assumed share of conversations that help.
threshold_results = []
for cutoff_percent in range(2, 91):
    cutoff = cutoff_percent / 100
    predicted_contact = validation["risk"] >= cutoff
    left = validation["left"] == 1
    true_positives = int((left & predicted_contact).sum())
    false_positives = int((~left & predicted_contact).sum())
    false_negatives = int((left & ~predicted_contact).sum())
    contacts = int(predicted_contact.sum())
    threshold_results.append(
        {
            "cutoff": cutoff,
            "contacts": contacts,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "net_value_dkk": (
                true_positives * conversation_help_rate * student_leaving_cost_dkk
                - contacts * conversation_cost_dkk
                - false_positives * false_alarm_cost_dkk
            ),
        }
    )
# Select the tested cutoff with the highest estimated net value.
best_value_rule = max(threshold_results, key=lambda result: result["net_value_dkk"])
# Count the errors and expected net value for the capacity-constrained rule too.
validation_left = validation["left"] == 1
capacity_false_positives = int(
    ((~validation_left) & validation["contacted"]).sum()
)
capacity_false_negatives = int(
    (validation_left & ~validation["contacted"]).sum()
)
capacity_true_positives = int((validation_left & validation["contacted"]).sum())
capacity_contacts = int(validation["contacted"].sum())
capacity_net_value_dkk = (
    capacity_true_positives * conversation_help_rate * student_leaving_cost_dkk
    - capacity_contacts * conversation_cost_dkk
    - capacity_false_positives * false_alarm_cost_dkk
)
expected_value_per_contact = (
    conversation_help_rate * student_leaving_cost_dkk + false_alarm_cost_dkk
)
break_even_cutoff = (
    (conversation_cost_dkk + false_alarm_cost_dkk) / expected_value_per_contact
    if expected_value_per_contact
    else 0.0
)

# Split the dashboard into the current list, a retrospective rule check, and group analysis.
this_week, validation_tab, groups_tab = st.tabs(
    ["This week's list", "2025 rule check", "Group comparison"]
)

with this_week:
    # Summarize how many current students fit the selected capacity and show the highest score.
    selected = current.loc[current["selected"]].sort_values("priority")
    first_metric, second_metric, third_metric = st.columns(3)
    first_metric.metric("Students this week", f"{len(current):,}")
    second_metric.metric("Conversations selected", f"{len(selected):,}")
    if len(selected):
        third_metric.metric("Highest predicted risk", f"{selected['risk'].max():.0%}")
    else:
        third_metric.metric("Highest predicted risk", "No contacts selected")

    # Show the ranked outreach list with week-6 fields useful for adviser review.
    table_columns = [
        column
        for column in [
            "priority",
            "student_id",
            "risk",
            "selected",
            "programme",
            "fees_owed",
            "submitted_share",
            "missed_last3",
            "logins_last3",
            "weeks_since_login",
        ]
        if column in current.columns
    ]
    ranked = current.sort_values("priority")[table_columns]
    st.dataframe(
        ranked,
        hide_index=True,
        width="stretch",
        column_config={
            "risk": st.column_config.ProgressColumn(
                "Predicted risk", format="percent", min_value=0, max_value=1
            ),
            "selected": st.column_config.CheckboxColumn("Within capacity"),
            "submitted_share": st.column_config.NumberColumn(
                "Submitted share", format="percent"
            ),
        },
    )
    st.download_button(
        "Download ranked list",
        data=ranked.to_csv(index=False),
        file_name="week6_ranked_students.csv",
        mime="text/csv",
    )

    if len(selected):
        # Let an adviser inspect one selected student's observed week-6 information.
        st.subheader("Student context for adviser review")
        student_ids = selected["student_id"].astype(str).tolist()
        chosen_id = st.selectbox("Select a student from the contact list", student_ids)
        student = selected.loc[selected["student_id"].astype(str) == chosen_id].iloc[0]
        st.write(
            f"Priority {int(student['priority'])} of {len(current)}; "
            f"estimated risk {student['risk']:.1%}."
        )
        context_labels = {
            "fees_owed": "Fees owed (week 6)",
            "submitted_share": "Assignments submitted (share)",
            "missed_last3": "Assignments missed in the last three weeks",
            "logins_last3": "Learning-platform logins in the last three weeks",
            "weeks_since_login": "Weeks since the most recent login",
            "quiz_mean": "Mean quiz score",
        }
        context = pd.DataFrame(
            [
                {"Week-6 information": label, "Observed value": student[column]}
                for column, label in context_labels.items()
                if column in student.index
            ]
        )
        st.dataframe(context, hide_index=True, width="stretch")
        st.caption(
            # These are observed inputs, not causal explanations for the score.
            "These observations provide context for a conversation; they do not explain why a student has a score."
        )

with validation_tab:
    # Compare the capacity-selected students with the outcomes that became known later in 2025.
    true_left = validation["left"].astype(bool)
    contacted = validation["contacted"].astype(bool)
    tp = int((true_left & contacted).sum())
    fp = int((~true_left & contacted).sum())
    fn = int((true_left & ~contacted).sum())
    tn = int((~true_left & ~contacted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    # Explain all four confusion-matrix groups and show precision/recall in plain language.
    st.subheader("What this capacity rule did on 2025")
    st.write(
        f"**{tp} students reached were later recorded as leaving**, "
        f"**{fp} contacted students stayed**, and **{fn} students who left were not selected**. "
        f"The remaining **{tn} students stayed and were not contacted**."
    )
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Precision", f"{precision:.1%}", help="Share of selected students who later left.")
    metric_two.metric("Recall", f"{recall:.1%}", help="Share of later leavers who were selected.")
    metric_three.metric("Validation students", f"{len(validation):,}")
    st.caption(
        "These are retrospective classifications, not proof that a conversation would prevent leaving."
    )

    # Compare the best tested threshold with the fixed-capacity rule using DKK net value.
    st.divider()
    st.subheader("Estimated value of the contact rule")
    st.caption(
        "A conversation costs the same whether the student later leaves or stays. A successful intervention is an assumed benefit; these estimates are not measured causal effects."
    )
    cost_one, cost_two, cost_three = st.columns(3)
    cost_one.metric(
        "Best tested cutoff",
        f"{best_value_rule['cutoff']:.0%}",
        delta=f"{best_value_rule['contacts']} contacts",
        delta_color="off",
    )
    cost_two.metric("Net value at cutoff", f"DKK {best_value_rule['net_value_dkk']:,.0f}")
    cost_three.metric("Net value under capacity", f"DKK {capacity_net_value_dkk:,.0f}")
    st.caption(
        f"Individual break-even risk is {break_even_cutoff:.1%}. "
        f"At capacity, {capacity_contacts} conversations cost DKK {capacity_contacts * conversation_cost_dkk:,.0f}; "
        f"there are {capacity_false_positives} false alarms and {capacity_false_negatives} missed leavers. "
        f"The best tested cutoff selects {best_value_rule['contacts']} students; "
        f"capacity is {contact_capacity}."
    )

with groups_tab:
    # Compare actual outcomes, mean scores, and recall separately for domestic and international students.
    st.subheader("2025 outcomes under the same capacity rule")
    rows = []
    for value, label in [(0, "Domestic"), (1, "International")]:
        group = validation.loc[validation["international"] == value]
        leavers = group["left"] == 1
        rows.append(
            {
                "Group": label,
                "Students": len(group),
                "Observed leave rate": group["left"].mean(),
                "Mean predicted risk": group["risk"].mean(),
                "Recall at capacity": group.loc[leavers, "contacted"].mean(),
            }
        )
    group_table = pd.DataFrame(rows)
    # Include historical login averages to make a possible group-level signal difference visible.
    login_means = history.groupby("international", observed=True)["logins_total"].mean()
    group_table["Mean logins (history)"] = [login_means.get(0), login_means.get(1)]
    st.dataframe(
        group_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Observed leave rate": st.column_config.NumberColumn(format="percent"),
            "Mean predicted risk": st.column_config.NumberColumn(format="percent"),
            "Recall at capacity": st.column_config.NumberColumn(format="percent"),
            "Mean logins (history)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "Group differences warrant review with more cohorts and affected students; do not use group membership as an automatic contact rule."
    )
