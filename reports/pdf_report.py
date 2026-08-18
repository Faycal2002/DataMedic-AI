"""
pdf_report.py

PDF report generation for the DataMedic AI project.

Automatically handles:

- Classification targets -> Target Balance
- Regression targets -> Target Distribution

Also includes:

- Dataset overview
- Data quality
- Machine Learning results
- Best model
- SHAP explainability
- SHAP visualization
- Conclusion
"""

import re
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _clean_report_name(report_name):
    """
    Clean a user-provided report name.
    """

    report_name = str(
        report_name
    ).strip()

    if report_name.lower().endswith(".pdf"):

        report_name = report_name[:-4]

    report_name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        report_name,
    )

    report_name = report_name.strip(
        " ."
    )

    if not report_name:

        report_name = (
            "dataset_doctor_report"
        )

    return report_name


def _get_unique_output_path(
    report_name,
    output_dir="reports/generated",
):
    """
    Create a unique PDF filename.
    """

    output_directory = Path(
        output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_name = _clean_report_name(
        report_name
    )

    output_path = (
        output_directory
        / f"{clean_name}.pdf"
    )

    counter = 2

    while output_path.exists():

        output_path = (
            output_directory
            / f"{clean_name}_{counter}.pdf"
        )

        counter += 1

    return output_path


def _build_conclusion(
    target_column,
    training_output,
    shap_output,
):
    """
    Create a short automatic conclusion.
    """

    problem_type = training_output[
        "problem_type"
    ]

    best_model_name = training_output[
        "best_model_name"
    ]

    primary_metric = training_output[
        "primary_metric"
    ]

    best_metrics = training_output[
        "results"
    ][best_model_name]["metrics"]

    best_score = best_metrics.get(
        primary_metric
    )

    top_features = (
        shap_output[
            "feature_importance"
        ]
        .head(3)["feature"]
        .tolist()
    )

    if best_score is not None:

        performance_text = (
            f"The best-performing model was "
            f"{best_model_name}, with a "
            f"{primary_metric.upper()} score of "
            f"{best_score:.4f}."
        )

    else:

        performance_text = (
            f"The best-performing model was "
            f"{best_model_name}."
        )

    if top_features:

        feature_text = (
            "The most influential features "
            "identified by SHAP were "
            + ", ".join(top_features)
            + "."
        )

    else:

        feature_text = (
            "SHAP analysis was completed "
            "for the selected model."
        )

    return (
        f"The selected target column was "
        f"'{target_column}'. "
        f"The task was identified as a "
        f"{problem_type} problem. "
        f"{performance_text} "
        f"{feature_text}"
    )


def _table_style():
    """
    Shared professional table style.
    """

    return TableStyle(
        [
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#eaf1ff"),
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.HexColor("#172033"),
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#d7ddea"),
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold",
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7,
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
        ]
    )


def generate_pdf_report(
    analysis,
    target_column,
    training_output,
    shap_output,
    dataset_name="Uploaded Dataset",
    report_name="dataset_doctor_report",
    output_path=None,
):
    """
    Generate the DataMedic AI PDF report.
    """

    # =========================================================
    # OUTPUT PATH
    # =========================================================

    if output_path is None:

        output_path = (
            _get_unique_output_path(
                report_name
            )
        )

        destination = str(
            output_path
        )

    elif output_path == ":memory:":

        pdf_buffer = BytesIO()

        destination = pdf_buffer

    else:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = str(
            output_path
        )

    # =========================================================
    # DOCUMENT
    # =========================================================

    document = SimpleDocTemplate(
        destination,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor(
            "#172033"
        ),
        spaceAfter=15,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor(
            "#172033"
        ),
        spaceBefore=10,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor(
            "#4f5d73"
        ),
    )

    story = []

    # =========================================================
    # TITLE
    # =========================================================

    display_report_name = (
        _clean_report_name(
            report_name
        )
    )

    story.append(
        Paragraph(
            display_report_name,
            title_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Dataset:</b> "
            f"{dataset_name}",
            normal_style,
        )
    )

    story.append(
        Spacer(1, 12)
    )

    # =========================================================
    # 1. DATASET OVERVIEW
    # =========================================================

    story.append(
        Paragraph(
            "1. Dataset Overview",
            heading_style,
        )
    )

    overview_data = [
        [
            "Item",
            "Value",
        ],

        [
            "Number of Rows",
            f"{analysis['n_rows']:,}",
        ],

        [
            "Number of Columns",
            str(
                analysis[
                    "n_columns"
                ]
            ),
        ],

        [
            "Missing Values",
            f"{analysis['n_missing']:,}",
        ],

        [
            "Duplicate Rows",
            f"{analysis['n_duplicates']:,}",
        ],

        [
            "Target Column",
            str(target_column),
        ],

        [
            "Problem Type",
            training_output[
                "problem_type"
            ].capitalize(),
        ],
    ]

    overview_table = Table(
        overview_data,
        colWidths=[
            7 * cm,
            9 * cm,
        ],
    )

    overview_table.setStyle(
        _table_style()
    )

    story.append(
        overview_table
    )

    story.append(
        Spacer(1, 14)
    )

    # =========================================================
    # 2. DATA QUALITY
    # =========================================================

    story.append(
        Paragraph(
            "2. Data Quality",
            heading_style,
        )
    )

    quality = analysis[
        "quality"
    ]

    quality_data = [
        [
            "Quality Measure",
            "Result",
        ],

        [
            "Overall Quality Score",
            f"{quality['score']} / 100",
        ],

        [
            "Missing Values",
            quality[
                "missing_status"
            ],
        ],

        [
            "Duplicate Rows",
            quality[
                "duplicate_status"
            ],
        ],

        [
            "Data Types",
            quality[
                "data_type_status"
            ],
        ],

        [
            "Correlation",
            quality[
                "correlation_status"
            ],
        ],

        [
            "Outliers",
            quality[
                "outlier_status"
            ],
        ],
    ]

    quality_table = Table(
        quality_data,
        colWidths=[
            8 * cm,
            8 * cm,
        ],
    )

    quality_table.setStyle(
        _table_style()
    )

    story.append(
        quality_table
    )

    story.append(
        Spacer(1, 14)
    )

    # =========================================================
    # 3. TARGET ANALYSIS
    # =========================================================

    problem_type = (
        training_output[
            "problem_type"
        ].lower()
    )

    # ---------------------------------------------------------
    # CLASSIFICATION
    # ---------------------------------------------------------

    if problem_type == "classification":

        story.append(
            Paragraph(
                "3. Target Balance",
                heading_style,
            )
        )

        target_balance = (
            analysis.get(
                "target_balance"
            )
        )

        if target_balance:

            if target_balance[
                "is_imbalanced"
            ]:

                balance_message = (
                    "The selected target "
                    "appears to be imbalanced."
                )

            else:

                balance_message = (
                    "The selected target "
                    "classes are reasonably "
                    "balanced."
                )

            story.append(
                Paragraph(
                    balance_message,
                    normal_style,
                )
            )

            proportions = (
                target_balance.get(
                    "class_proportions",
                    {},
                )
            )

            if proportions:

                balance_data = [
                    [
                        "Class",
                        "Proportion",
                    ]
                ]

                for (
                    class_name,
                    proportion,
                ) in proportions.items():

                    balance_data.append(
                        [
                            str(
                                class_name
                            ),
                            f"{proportion:.2%}",
                        ]
                    )

                balance_table = Table(
                    balance_data,
                    colWidths=[
                        8 * cm,
                        8 * cm,
                    ],
                )

                balance_table.setStyle(
                    _table_style()
                )

                story.append(
                    Spacer(1, 8)
                )

                story.append(
                    balance_table
                )

    # ---------------------------------------------------------
    # REGRESSION
    # ---------------------------------------------------------

    else:

        story.append(
            Paragraph(
                "3. Target Distribution",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                (
                    f"The selected target "
                    f"'<b>{target_column}</b>' "
                    "is treated as a continuous "
                    "numeric variable. "
                    "The following statistics "
                    "summarize its distribution."
                ),
                normal_style,
            )
        )

        story.append(
            Spacer(1, 8)
        )

        target_distribution = (
            analysis.get(
                "target_distribution"
            )
        )

        if target_distribution:

            td = target_distribution

            def format_value(value):

                if value is None:
                    return "-"

                return f"{value:,.2f}"

            distribution_data = [
                [
                    "Statistic",
                    "Value",
                ],

                [
                    "Mean",
                    format_value(
                        td.get("mean")
                    ),
                ],

                [
                    "Median",
                    format_value(
                        td.get("median")
                    ),
                ],

                [
                    "Minimum",
                    format_value(
                        td.get("minimum")
                    ),
                ],

                [
                    "Maximum",
                    format_value(
                        td.get("maximum")
                    ),
                ],

                [
                    "Standard Deviation",
                    format_value(
                        td.get("std")
                    ),
                ],
            ]

            distribution_table = Table(
                distribution_data,
                colWidths=[
                    8 * cm,
                    8 * cm,
                ],
            )

            distribution_table.setStyle(
                _table_style()
            )

            story.append(
                distribution_table
            )

        else:

            story.append(
                Paragraph(
                    "No target distribution "
                    "statistics are available.",
                    normal_style,
                )
            )

    story.append(
        Spacer(1, 14)
    )

    # =========================================================
    # 4. MACHINE LEARNING RESULTS
    # =========================================================

    story.append(
        Paragraph(
            "4. Machine Learning Results",
            heading_style,
        )
    )

    model_results = (
        training_output[
            "results"
        ]
    )

    metric_names = []

    for model_info in (
        model_results.values()
    ):

        for metric in (
            model_info[
                "metrics"
            ]
        ):

            if metric not in metric_names:

                metric_names.append(
                    metric
                )

    model_table_data = [
        ["Model"]
        + [
            metric.upper()
            for metric in metric_names
        ]
    ]

    for (
        model_name,
        model_info,
    ) in model_results.items():

        row = [
            model_name
        ]

        for metric in metric_names:

            metric_value = (
                model_info[
                    "metrics"
                ].get(metric)
            )

            if metric_value is None:

                row.append("-")

            else:

                row.append(
                    f"{metric_value:.4f}"
                )

        model_table_data.append(
            row
        )

    available_width = 16 * cm

    column_width = (
        available_width
        / len(
            model_table_data[0]
        )
    )

    model_table = Table(
        model_table_data,
        colWidths=[
            column_width
            for _ in model_table_data[0]
        ],
    )

    model_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#eaf1ff"
                    ),
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#d7ddea"
                    ),
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(
        model_table
    )

    story.append(
        Spacer(1, 12)
    )

    # =========================================================
    # BEST MODEL
    # =========================================================

    best_model_name = (
        training_output[
            "best_model_name"
        ]
    )

    primary_metric = (
        training_output[
            "primary_metric"
        ]
    )

    best_score = (
        training_output[
            "results"
        ][best_model_name][
            "metrics"
        ].get(
            primary_metric
        )
    )

    best_model_text = (
        f"<b>Best Model:</b> "
        f"{best_model_name}"
    )

    if best_score is not None:

        best_model_text += (
            f"<br/><b>"
            f"{primary_metric.upper()}:"
            f"</b> "
            f"{best_score:.4f}"
        )

    story.append(
        Paragraph(
            best_model_text,
            normal_style,
        )
    )

    story.append(
        Spacer(1, 14)
    )

    # =========================================================
    # 5. SHAP
    # =========================================================

    story.append(
        Paragraph(
            "5. Explainable AI / SHAP",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            (
                "SHAP was used to explain "
                "which features had the strongest "
                "influence on the predictions made "
                "by the selected best-performing model."
            ),
            normal_style,
        )
    )

    story.append(
        Spacer(1, 8)
    )

    feature_importance = (
        shap_output[
            "feature_importance"
        ]
        .head(10)
    )

    shap_table_data = [
        [
            "Rank",
            "Feature",
            "Mean |SHAP Value|",
        ]
    ]

    for rank, (
        index,
        row,
    ) in enumerate(
        feature_importance.iterrows(),
        start=1,
    ):

        shap_table_data.append(
            [
                str(rank),
                str(
                    row["feature"]
                ),
                f"{row['importance']:.4f}",
            ]
        )

    shap_table = Table(
        shap_table_data,
        colWidths=[
            2 * cm,
            10 * cm,
            4 * cm,
        ],
    )

    shap_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#eaf1ff"
                    ),
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#d7ddea"
                    ),
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(
        shap_table
    )

    story.append(
        Spacer(1, 14)
    )

    # =========================================================
    # SHAP IMAGE
    # =========================================================

    shap_plot_path = (
        shap_output.get(
            "summary_plot_path"
        )
    )

    if shap_plot_path:

        plot_file = Path(
            shap_plot_path
        )

        if plot_file.exists():

            story.append(
                Paragraph(
                    "SHAP Summary Visualization",
                    heading_style,
                )
            )

            shap_image = Image(
                str(
                    plot_file
                ),
                width=16 * cm,
                height=9 * cm,
            )

            story.append(
                shap_image
            )

            story.append(
                Spacer(1, 14)
            )

    # =========================================================
    # 6. CONCLUSION
    # =========================================================

    story.append(
        Paragraph(
            "6. Conclusion",
            heading_style,
        )
    )

    conclusion = _build_conclusion(
        target_column,
        training_output,
        shap_output,
    )

    story.append(
        Paragraph(
            conclusion,
            normal_style,
        )
    )

    # =========================================================
    # BUILD PDF
    # =========================================================

    document.build(
        story
    )

    if output_path == ":memory:":

        pdf_buffer.seek(0)

        return pdf_buffer

    return str(
        output_path
    )