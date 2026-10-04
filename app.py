from pathlib import Path

import pandas as pd
import streamlit as st

import portable


DATA_URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"
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

st.set_page_config(page_title="Study Office | Week 6", layout="wide")
st.title("Study Office | Week 6")
st.caption(
    "A human-reviewed support-priority list. A risk score is not a decision or a diagnosis."
)


@st.cache_data
def load_data():
    history = pd.read_csv(f"{DATA_URL}history_week6.csv")
    current = pd.read_csv(f"{DATA_URL}new_week6.csv")
    return history, current


@st.cache_resource
def load_model():
    return portable.Model(str(Path(__file__).parent / "model"))


def add_risk(frame, model):
    scored = frame.copy()
    scored["risk"] = pd.Series(
        model.predict_proba(scored[FEATURES]), index=scored.index, dtype="float64"
    )
    scored["priority"] = scored["risk"].rank(method="first", ascending=False).astype(int)
    return scored


try:
    history, current = load_data()
    model = load_model()
    validation = add_risk(history.loc[history["cohort"] == 2025].copy(), model)
    current = add_risk(current, model)
except Exception as error:
    st.error(f"The data or model could not be loaded: {error}")
    st.stop()

max_contacts = max(40, len(current))
st.sidebar.subheader("Error-cost assumptions")
false_alarm_cost_eur = st.sidebar.number_input(
    "False alarm cost (EUR)", min_value=0, value=10, step=5
)
missed_student_cost_eur = st.sidebar.number_input(
    "Missed student cost (EUR)", min_value=0, value=100, step=10
)
default_capacity = min(40, max_contacts)
st.session_state.setdefault("capacity_slider", default_capacity)
st.session_state.setdefault("capacity_input", default_capacity)
st.session_state["capacity_slider"] = min(
    max_contacts, max(0, st.session_state["capacity_slider"])
)
st.session_state["capacity_input"] = min(
    max_contacts, max(0, st.session_state["capacity_input"])
)
st.session_state["capacity_input"] = st.session_state["capacity_slider"]


def sync_capacity_input():
    st.session_state["capacity_input"] = st.session_state["capacity_slider"]


def sync_capacity_slider():
    st.session_state["capacity_slider"] = st.session_state["capacity_input"]


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

current["selected"] = current["priority"] <= contact_capacity
validation["contacted"] = validation["priority"] <= min(contact_capacity, len(validation))

threshold_results = []
for cutoff_percent in range(2, 91):
    cutoff = cutoff_percent / 100
    predicted_contact = validation["risk"] >= cutoff
    false_positives = int(((validation["left"] == 0) & predicted_contact).sum())
    false_negatives = int(((validation["left"] == 1) & ~predicted_contact).sum())
    threshold_results.append(
        {
            "cutoff": cutoff,
            "contacts": int(predicted_contact.sum()),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "cost_eur": false_positives * false_alarm_cost_eur
            + false_negatives * missed_student_cost_eur,
        }
    )
best_cost_rule = min(threshold_results, key=lambda result: result["cost_eur"])
capacity_false_positives = int(
    ((validation["left"] == 0) & validation["contacted"]).sum()
)
capacity_false_negatives = int(
    ((validation["left"] == 1) & ~validation["contacted"]).sum()
)
capacity_error_cost_eur = (
    capacity_false_positives * false_alarm_cost_eur
    + capacity_false_negatives * missed_student_cost_eur
)
total_error_cost_eur = false_alarm_cost_eur + missed_student_cost_eur
break_even_cutoff = (
    false_alarm_cost_eur / total_error_cost_eur if total_error_cost_eur else 0.0
)

this_week, validation_tab, groups_tab = st.tabs(
    ["This week's list", "2025 rule check", "Group comparison"]
)

with this_week:
    selected = current.loc[current["selected"]].sort_values("priority")
    first_metric, second_metric, third_metric = st.columns(3)
    first_metric.metric("Students this week", f"{len(current):,}")
    second_metric.metric("Conversations selected", f"{len(selected):,}")
    if len(selected):
        third_metric.metric("Highest predicted risk", f"{selected['risk'].max():.0%}")
    else:
        third_metric.metric("Highest predicted risk", "No contacts selected")

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
            "These observations provide context for a conversation; they do not explain why a student has a score."
        )

with validation_tab:
    true_left = validation["left"].astype(bool)
    contacted = validation["contacted"].astype(bool)
    tp = int((true_left & contacted).sum())
    fp = int((~true_left & contacted).sum())
    fn = int((true_left & ~contacted).sum())
    tn = int((~true_left & ~contacted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

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

    st.divider()
    st.subheader("Your error-cost scenario")
    st.caption(
        "Edit the EUR costs in the sidebar. The rule below minimizes false-positive and false-negative costs on 2025; it does not include a separate contact cost or intervention benefit."
    )
    cost_one, cost_two, cost_three = st.columns(3)
    cost_one.metric(
        "Lowest-cost validation cutoff",
        f"{best_cost_rule['cutoff']:.0%}",
        delta=f"{best_cost_rule['contacts']} contacts",
        delta_color="off",
    )
    cost_two.metric("Error cost at cutoff", f"EUR {best_cost_rule['cost_eur']:,.0f}")
    cost_three.metric("Cost under capacity rule", f"EUR {capacity_error_cost_eur:,.0f}")
    st.caption(
        f"Cost-ratio break-even is {break_even_cutoff:.1%}. "
        f"At the selected capacity there are {capacity_false_positives} false alarms and "
        f"{capacity_false_negatives} missed leavers. "
        f"The cost-minimizing cutoff selects {best_cost_rule['contacts']} students; "
        f"capacity is {contact_capacity}."
    )

with groups_tab:
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
