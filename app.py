import io
import textwrap

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="SkillPath AI",
    page_icon="🧭",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM DESIGN
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, #eef2ff, transparent 35%),
            linear-gradient(135deg, #f8fafc, #f5f3ff);
    }

    .hero {
        padding: 2.3rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(135deg, #312e81, #6366f1, #06b6d4);
        box-shadow: 0 18px 45px rgba(49, 46, 129, 0.22);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 3rem;
    }

    .hero p {
        margin-top: 0.8rem;
        font-size: 1.1rem;
        opacity: 0.95;
    }

    .result-card {
        padding: 1.3rem;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.90);
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.07);
        margin-bottom: 1rem;
    }

    .next-card {
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        background: linear-gradient(135deg, #f59e0b, #f97316);
        box-shadow: 0 12px 30px rgba(249, 115, 22, 0.24);
        margin-bottom: 1rem;
    }

    .roadmap-completed {
        padding: 1rem;
        margin: 0.6rem 0;
        border-radius: 14px;
        border-left: 6px solid #10b981;
        background: #ecfdf5;
    }

    .roadmap-next {
        padding: 1rem;
        margin: 0.6rem 0;
        border-radius: 14px;
        border-left: 6px solid #f59e0b;
        background: #fffbeb;
    }

    .roadmap-upcoming {
        padding: 1rem;
        margin: 0.6rem 0;
        border-radius: 14px;
        border-left: 6px solid #6366f1;
        background: #eef2ff;
    }

    .small-label {
        color: #64748b;
        font-size: 0.88rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 16px;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD TRAINED MODELS
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    readiness = joblib.load("readiness_model.pkl")

    recommender = joblib.load(
        "next_skill_classifier.pkl"
    )

    return readiness, recommender


try:
    readiness_model, next_skill_model = load_models()

except FileNotFoundError:
    st.error(
        "The trained model files were not found. "
        "Make sure readiness_model.pkl and "
        "next_skill_classifier.pkl are in the same "
        "folder as app.py."
    )

    st.stop()


# ---------------------------------------------------------
# SKILL DEPENDENCY PATHS
# ---------------------------------------------------------

learning_paths = {
    "Data Science": [
        "Python Fundamentals",
        "Mathematics for Data Science",
        "Statistics Fundamentals",
        "SQL and Data Handling",
        "Machine Learning Fundamentals",
        "Model Evaluation",
        "Deep Learning",
        "Data Science Capstone"
    ],

    "Artificial Intelligence": [
        "Python Fundamentals",
        "Mathematics for AI",
        "Probability and Statistics",
        "Machine Learning Fundamentals",
        "Neural Networks",
        "Deep Learning",
        "Generative AI",
        "AI Capstone"
    ],

    "Web Development": [
        "HTML and CSS",
        "JavaScript Fundamentals",
        "Database and SQL",
        "Backend Development",
        "React and API Development",
        "Full-Stack Testing",
        "Deployment",
        "Full-Stack Capstone"
    ]
}


domain_skill_fields = {
    "Data Science": {
        "Python": "python_score",
        "Mathematics": "mathematics_score",
        "Statistics": "statistics_score",
        "Database": "database_score",
        "Logical Reasoning": "logical_reasoning_score"
    },

    "Artificial Intelligence": {
        "Python": "python_score",
        "Mathematics": "mathematics_score",
        "Statistics": "statistics_score",
        "Database": "database_score",
        "Logical Reasoning": "logical_reasoning_score"
    },

    "Web Development": {
        "HTML/CSS": "html_css_score",
        "JavaScript": "javascript_score",
        "Database": "database_score",
        "Python": "python_score",
        "Logical Reasoning": "logical_reasoning_score"
    }
}


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def readiness_level(score):

    if score < 40:
        return "Foundation Required"

    if score < 60:
        return "Developing"

    if score < 80:
        return "Almost Ready"

    return "Ready"


def readiness_colour(score):

    if score < 40:
        return "#ef4444"

    if score < 60:
        return "#f59e0b"

    if score < 80:
        return "#06b6d4"

    return "#10b981"


def build_gauge(score):

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            title={"text": "Predicted Readiness"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": readiness_colour(score)
                },
                "steps": [
                    {
                        "range": [0, 40],
                        "color": "#fee2e2"
                    },
                    {
                        "range": [40, 60],
                        "color": "#fef3c7"
                    },
                    {
                        "range": [60, 80],
                        "color": "#cffafe"
                    },
                    {
                        "range": [80, 100],
                        "color": "#d1fae5"
                    }
                ],
                "threshold": {
                    "line": {
                        "color": "#111827",
                        "width": 4
                    },
                    "value": score
                }
            }
        )
    )

    figure.update_layout(
        height=320,
        margin=dict(l=25, r=25, t=65, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"}
    )

    return figure


def create_pdf(report):

    buffer = io.BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=A4
    )

    width, height = A4
    y = height - 60

    pdf.setTitle("SkillPath AI Learning Report")

    pdf.setFont("Helvetica-Bold", 19)

    pdf.drawString(
        50,
        y,
        "SkillPath AI - Personalized Learning Report"
    )

    y -= 42
    pdf.setFont("Helvetica", 11)

    lines = [
        f"Student: {report['student_name']}",
        f"Target Domain: {report['target_domain']}",
        f"Predicted Readiness: {report['readiness_score']}%",
        f"Readiness Level: {report['readiness_level']}",
        f"Recommended Next Skill: {report['next_skill']}",
        f"Model Confidence: {report['confidence']}%",
        f"Strongest Skill: {report['strongest_skill']}",
        f"Critical Skill Gap: {report['critical_gap']}",
        "",
        "Personalized Remaining Learning Path:"
    ]

    for line in lines:
        pdf.drawString(50, y, line)
        y -= 22

    for number, skill in enumerate(
        report["remaining_path"],
        start=1
    ):
        wrapped_lines = textwrap.wrap(
            f"{number}. {skill}",
            width=75
        )

        for line in wrapped_lines:
            pdf.drawString(65, y, line)
            y -= 20

    y -= 12

    explanation = (
        f"{report['next_skill']} was recommended because it is "
        f"the earliest unresolved prerequisite detected for the "
        f"student's {report['target_domain']} goal."
    )

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, y, "Explanation:")

    y -= 20
    pdf.setFont("Helvetica", 10)

    for line in textwrap.wrap(explanation, width=85):
        pdf.drawString(50, y, line)
        y -= 17

    pdf.save()
    buffer.seek(0)

    return buffer.getvalue()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("🧭 SkillPath AI")

    page = st.radio(
        "Navigate",
        [
            "Skill Assessment",
            "How It Works",
            "Model Details"
        ]
    )

    st.divider()

    st.caption(
        "Dependency-aware learning paths powered by "
        "regression, classification and explainable AI."
    )


# ---------------------------------------------------------
# ASSESSMENT PAGE
# ---------------------------------------------------------

if page == "Skill Assessment":

    st.markdown(
        """
        <div class="hero">
            <h1>SkillPath AI</h1>
            <p>
                Discover your readiness, uncover the prerequisite
                holding you back, and receive a learning path built
                specifically for your goal.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Student Diagnostic Assessment")

    with st.form("skill_assessment_form"):

        first, second = st.columns(2)

        with first:
            student_name = st.text_input(
                "Student name",
                placeholder="Enter your name"
            )

        with second:
            target_domain = st.selectbox(
                "Target career domain",
                list(learning_paths.keys())
            )

        st.markdown("### Technical Skill Scores")

        column1, column2, column3 = st.columns(3)

        with column1:
            python_score = st.slider(
                "Python",
                0,
                100,
                50
            )

            mathematics_score = st.slider(
                "Mathematics",
                0,
                100,
                50
            )

            statistics_score = st.slider(
                "Statistics",
                0,
                100,
                50
            )

        with column2:
            html_css_score = st.slider(
                "HTML and CSS",
                0,
                100,
                40
            )

            javascript_score = st.slider(
                "JavaScript",
                0,
                100,
                40
            )

            database_score = st.slider(
                "Database and SQL",
                0,
                100,
                50
            )

        with column3:
            logical_reasoning_score = st.slider(
                "Logical reasoning",
                0,
                100,
                50
            )

            assessment_score = st.slider(
                "Diagnostic assessment score",
                0,
                100,
                50
            )

            weekly_hours = st.slider(
                "Available hours per week",
                1,
                20,
                8
            )

        st.markdown("### Learning Experience")

        experience1, experience2 = st.columns(2)

        with experience1:
            completed_courses = st.number_input(
                "Completed relevant courses",
                min_value=0,
                max_value=10,
                value=2
            )

        with experience2:
            experience_months = st.number_input(
                "Practical experience in months",
                min_value=0,
                max_value=24,
                value=3
            )

        submitted = st.form_submit_button(
            "Generate My Learning Path",
            use_container_width=True,
            type="primary"
        )

    if submitted:

        if not student_name.strip():
            student_name = "Student"

        student = {
            "target_domain": target_domain,
            "python_score": python_score,
            "mathematics_score": mathematics_score,
            "statistics_score": statistics_score,
            "html_css_score": html_css_score,
            "javascript_score": javascript_score,
            "database_score": database_score,
            "logical_reasoning_score": logical_reasoning_score,
            "assessment_score": assessment_score,
            "weekly_hours": weekly_hours,
            "completed_courses": completed_courses,
            "experience_months": experience_months
        }

        student_frame = pd.DataFrame([student])

        predicted_readiness = readiness_model.predict(
            student_frame
        )[0]

        predicted_readiness = float(
            np.clip(predicted_readiness, 0, 100)
        )

        level = readiness_level(
            predicted_readiness
        )

        recommendation_input = student_frame.copy()

        recommendation_input["readiness_score"] = (
            predicted_readiness
        )

        predicted_skill = next_skill_model.predict(
            recommendation_input
        )[0]

        probability = next_skill_model.predict_proba(
            recommendation_input
        )[0]

        confidence = float(np.max(probability) * 100)

        complete_path = learning_paths[target_domain]

        if predicted_skill in complete_path:
            current_index = complete_path.index(
                predicted_skill
            )
        else:
            current_index = 0

        completed_path = complete_path[:current_index]
        remaining_path = complete_path[current_index:]

        skill_mapping = domain_skill_fields[
            target_domain
        ]

        skill_scores = {
            label: student[field]
            for label, field in skill_mapping.items()
        }

        sorted_skills = sorted(
            skill_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        strongest_skill = sorted_skills[0]
        critical_gap = sorted_skills[-1]

        st.divider()
        st.header(f"{student_name}'s Learning Intelligence Report")

        gauge_column, insight_column = st.columns(
            [1.1, 1]
        )

        with gauge_column:
            st.plotly_chart(
                build_gauge(predicted_readiness),
                use_container_width=True
            )

        with insight_column:

            st.markdown(
                f"""
                <div class="next-card">
                    <div class="small-label"
                         style="color:#ffedd5;">
                        RECOMMENDED NEXT SKILL
                    </div>
                    <h2>{predicted_skill}</h2>
                    <p>
                        Model confidence: {confidence:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            metric1, metric2 = st.columns(2)

            metric1.metric(
                "Readiness level",
                level
            )

            metric2.metric(
                "Target domain",
                target_domain
            )

        st.subheader("Diagnostic Insights")

        insight1, insight2 = st.columns(2)

        with insight1:
            st.success(
                f"Strongest skill: "
                f"{strongest_skill[0]} "
                f"({strongest_skill[1]}%)"
            )

        with insight2:
            st.warning(
                f"Critical skill gap: "
                f"{critical_gap[0]} "
                f"({critical_gap[1]}%)"
            )

        st.info(
            f"Why this recommendation? {predicted_skill} is "
            f"the earliest unresolved prerequisite detected "
            f"for your {target_domain} goal. Strengthening it "
            f"reduces the risk of struggling with later topics."
        )

        st.subheader("Personalized Skill Dependency Path")

        for index, skill in enumerate(complete_path):

            if index < current_index:
                css_class = "roadmap-completed"
                status = "✓ Completed prerequisite"

            elif index == current_index:
                css_class = "roadmap-next"
                status = "→ Learn this next"

            else:
                css_class = "roadmap-upcoming"
                status = "Upcoming"

            st.markdown(
                f"""
                <div class="{css_class}">
                    <strong>Step {index + 1}: {skill}</strong>
                    <br>
                    <span class="small-label">{status}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        report = {
            "student_name": student_name,
            "target_domain": target_domain,
            "readiness_score": round(
                predicted_readiness,
                2
            ),
            "readiness_level": level,
            "next_skill": predicted_skill,
            "confidence": round(confidence, 2),
            "strongest_skill": (
                f"{strongest_skill[0]} "
                f"({strongest_skill[1]}%)"
            ),
            "critical_gap": (
                f"{critical_gap[0]} "
                f"({critical_gap[1]}%)"
            ),
            "remaining_path": remaining_path
        }

        pdf_data = create_pdf(report)

        st.download_button(
            "Download Personalized PDF Report",
            data=pdf_data,
            file_name=(
                f"{student_name.replace(' ', '_')}"
                "_skillpath_report.pdf"
            ),
            mime="application/pdf",
            use_container_width=True
        )


# ---------------------------------------------------------
# HOW IT WORKS PAGE
# ---------------------------------------------------------

elif page == "How It Works":

    st.title("How SkillPath AI Works")

    st.markdown(
        """
        ### 1. Diagnostic profile

        The student selects a target domain and provides
        assessment, skill and learning-experience information.

        ### 2. Readiness prediction

        Module 1 uses regression to predict a readiness score
        between 0 and 100.

        ### 3. Next-skill classification

        Module 2 combines the student profile with the predicted
        readiness score to classify the most appropriate next skill.

        ### 4. Dependency-aware roadmap

        The predicted skill is positioned inside an expert-defined
        prerequisite graph. Completed and upcoming skills are then
        arranged in the correct order.

        ### 5. Explainable output

        The system displays the strongest skill, critical skill gap
        and a reason for the recommendation.
        """
    )


# ---------------------------------------------------------
# MODEL DETAILS PAGE
# ---------------------------------------------------------

else:

    st.title("Machine Learning Model Details")

    model1, model2 = st.columns(2)

    with model1:

        st.markdown(
            """
            ### Module 1: Readiness Prediction

            **Learning type:** Supervised regression

            **Models compared:**

            - Dummy baseline
            - Multiple Linear Regression
            - Polynomial Regression

            **Evaluation:**

            - MAE
            - RMSE
            - R² score
            - Five-fold cross-validation
            - Residual analysis
            - Permutation importance
            """
        )

    with model2:

        st.markdown(
            """
            ### Module 2: Next-Skill Recommendation

            **Learning type:** Supervised classification

            **Models compared:**

            - Dummy baseline
            - Decision Tree
            - Random Forest

            **Evaluation:**

            - Accuracy
            - Macro precision
            - Macro recall
            - Macro F1-score
            - Five-fold cross-validation
            - Confusion matrix
            """
        )

    st.warning(
        "Prototype limitation: the current models are validated "
        "using transparent synthetic and expert-rule-labelled data. "
        "Real student and academic-advisor data is required before "
        "institutional deployment."
    )