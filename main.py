import os
import re
import requests
import pandas as pd

from agents import Agent, Runner, function_tool
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel


# ============================================================
# 1. CONNECT TO GEMINI
# ============================================================

gemini_client = AsyncOpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-3.6-flash",
    openai_client=gemini_client,
)


# ============================================================
# 2. COMMON DATA QUALITY ANALYSIS
# ============================================================

def analyze_dataframe(df: pd.DataFrame) -> str:

    report = []

    # Dataset overview
    report.append("DATASET OVERVIEW")
    report.append(f"Number of rows: {len(df)}")
    report.append(f"Number of columns: {len(df.columns)}")
    report.append(
        f"Column names: {', '.join(df.columns.astype(str))}"
    )

    # Missing values
    report.append("\nMISSING VALUES")

    missing_count = df.isnull().sum()
    missing_percentage = (
        df.isnull().mean() * 100
    ).round(2)

    for column in df.columns:

        count = missing_count[column]
        percentage = missing_percentage[column]

        report.append(
            f"{column}: {count} missing ({percentage}%)"
        )

    # Duplicate rows
    report.append("\nDUPLICATE ROWS")

    duplicates = df.duplicated().sum()

    report.append(
        f"Duplicate rows: {duplicates}"
    )

    # Data types
    report.append("\nDATA TYPES")

    for column in df.columns:

        report.append(
            f"{column}: {df[column].dtype}"
        )

    # Unique values
    report.append("\nUNIQUE VALUES")

    for column in df.columns:

        unique_count = df[column].nunique(
            dropna=True
        )

        report.append(
            f"{column}: {unique_count} unique values"
        )

    # Numeric statistics
    report.append(
        "\nNUMERIC COLUMN STATISTICS"
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) == 0:

        report.append(
            "No numeric columns found."
        )

    else:

        for column in numeric_columns:

            report.append(
                f"\n{column}:"
            )

            report.append(
                f"  Minimum: {df[column].min()}"
            )

            report.append(
                f"  Maximum: {df[column].max()}"
            )

            report.append(
                f"  Mean: {df[column].mean():.2f}"
            )

            report.append(
                f"  Median: {df[column].median():.2f}"
            )

    # Outlier detection
    report.append(
        "\nANOMALY / OUTLIER DETECTION"
    )

    found_outliers = False

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 4:
            continue

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        ]

        if len(outliers) > 0:

            found_outliers = True

            report.append(
                f"\n{column}:"
            )

            report.append(
                f"  Lower bound: {lower_bound:.2f}"
            )

            report.append(
                f"  Upper bound: {upper_bound:.2f}"
            )

            report.append(
                f"  Outlier count: {len(outliers)}"
            )

            report.append(
                f"  Outlier values: "
                f"{outliers[column].tolist()}"
            )

    if not found_outliers:

        report.append(
            "No statistical outliers detected."
        )

    # Age validation
    if "age" in df.columns:

        report.append(
            "\nAGE VALIDATION"
        )

        numeric_age = pd.to_numeric(
            df["age"],
            errors="coerce"
        )

        invalid_age = (
            numeric_age.isna()
            & df["age"].notna()
        ).sum()

        unrealistic_age = (
            (numeric_age < 0)
            | (numeric_age > 120)
        ).sum()

        report.append(
            f"Invalid age values: {invalid_age}"
        )

        report.append(
            f"Unrealistic age values: "
            f"{unrealistic_age}"
        )

    # Email validation
    if "email" in df.columns:

        report.append(
            "\nEMAIL VALIDATION"
        )

        email_pattern = (
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
        )

        invalid_emails = (
            df["email"]
            .dropna()
            .astype(str)
            .apply(
                lambda x: not bool(
                    re.match(
                        email_pattern,
                        x
                    )
                )
            )
            .sum()
        )

        report.append(
            f"Invalid email addresses: "
            f"{invalid_emails}"
        )

    return "\n".join(report)


# ============================================================
# 3. CSV QUALITY TOOL
# ============================================================

def check_csv_quality(file_path: str) -> str:

    try:

        df = pd.read_csv(file_path)

        return analyze_dataframe(df)

    except Exception as e:

        return (
            f"Could not analyze the CSV file.\n"
            f"Error: {str(e)}"
        )


# ============================================================
# 4. PUBLIC API TOOL
# ============================================================

def get_api_data() -> str:

    try:

        url = "https://fakestoreapi.com/products"

        response = requests.get(
            url,
            timeout=10
        )

        if response.status_code != 200:

            return (
                "API request failed.\n"
                f"Status code: "
                f"{response.status_code}"
            )

        data = response.json()

        df = pd.DataFrame(data)

        return analyze_dataframe(df)

    except Exception as e:

        return (
            f"Could not retrieve or analyze "
            f"the API data.\n"
            f"Error: {str(e)}"
        )


# ============================================================
# 5. DATA CLEANING TOOL
# ============================================================

def clean_csv(file_path: str) -> str:
    """
    Create a cleaned copy of a CSV file.

    The original file is not modified.
    """

    try:

        # Load original CSV
        df = pd.read_csv(file_path)

        original_rows = len(df)

        # ----------------------------------------------------
        # Remove exact duplicate rows
        # ----------------------------------------------------

        df = df.drop_duplicates()

        duplicates_removed = (
            original_rows - len(df)
        )

        # ----------------------------------------------------
        # Clean age column
        # ----------------------------------------------------

        invalid_age_values = 0

        if "age" in df.columns:

            original_age = df["age"].copy()

            df["age"] = pd.to_numeric(
                df["age"],
                errors="coerce"
            )

            invalid_age_values = (
                df["age"].isna()
                & original_age.notna()
            ).sum()

        # ----------------------------------------------------
        # Create output filename
        # ----------------------------------------------------

        base_name = os.path.basename(
            file_path
        )

        file_name, extension = os.path.splitext(
            base_name
        )

        output_file = (
            f"cleaned_{file_name}{extension}"
        )

        # ----------------------------------------------------
        # Save cleaned CSV
        # ----------------------------------------------------

        df.to_csv(
            output_file,
            index=False
        )

        # ----------------------------------------------------
        # Return cleaning summary
        # ----------------------------------------------------

        return (
            "DATA CLEANING COMPLETED\n\n"
            f"Original rows: {original_rows}\n"
            f"Final rows: {len(df)}\n"
            f"Duplicate rows removed: "
            f"{duplicates_removed}\n"
            f"Invalid age values converted "
            f"to missing: {invalid_age_values}\n"
            f"Cleaned file: {output_file}"
        )

    except Exception as e:

        return (
            "Could not clean the CSV file.\n"
            f"Error: {str(e)}"
        )


# ============================================================
# 6. CONVERT FUNCTIONS INTO AGENT TOOLS
# ============================================================

csv_quality_tool = function_tool(
    check_csv_quality
)

api_data_tool = function_tool(
    get_api_data
)

clean_csv_tool = function_tool(
    clean_csv
)


# ============================================================
# 7. CREATE THE AI AGENT
# ============================================================

agent = Agent(

    name="Data Quality Agent",

    instructions="""
    You are an AI Data Quality Agent.

    Your job is to analyze datasets,
    identify data quality problems,
    and help clean CSV files.

    You have three tools.

    TOOL 1:
    check_csv_quality

    Use this when the user provides a CSV
    file and wants it analyzed.

    TOOL 2:
    get_api_data

    Use this when the user asks you to
    retrieve or analyze product data from
    the public FakeStoreAPI.

    TOOL 3:
    clean_csv

    Use this ONLY when the user explicitly
    asks you to clean or fix a CSV file.

    Never modify the original CSV file.

    The cleaning tool creates a new file
    beginning with "cleaned_".

    When analyzing data, explain:

    - Dataset size
    - Missing values
    - Duplicate rows
    - Data types
    - Unique values
    - Numeric statistics
    - Invalid values
    - Statistical outliers
    - Potential anomalies

    Clearly distinguish between:

    1. Confirmed data-quality problems
    2. Statistical outliers that require investigation

    When cleaning data, explain exactly
    what was changed.

    Keep reports clear and organized.
    """,

    model=gemini_model,

    tools=[
        csv_quality_tool,
        api_data_tool,
        clean_csv_tool
    ]
)


# ============================================================
# 8. ASK THE USER WHAT TO DO
# ============================================================

user_request = input(
    "\nWhat would you like me to do?\n> "
)


# ============================================================
# 9. RUN THE AGENT
# ============================================================

result = Runner.run_sync(
    agent,
    user_request
)


# ============================================================
# 10. DISPLAY RESULT
# ============================================================

print("\n")
print("========================================")
print("       DATA QUALITY AGENT")
print("========================================")
print("\n")

print(result.final_output)