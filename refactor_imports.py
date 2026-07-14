import os
import re


def update_imports(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Files moved to utils/
    utils_modules = [
        "pdf_generator",
        "gradcam",
        "leaf_validator",
        "logger_config",
        "exceptions",
    ]
    # Files moved to services/
    services_modules = [
        "gemini_advisor",
        "llm_manager",
        "ai_advisor",
        "verify_gemini",
        "prediction_logger",
    ]
    # Files moved to streamlit/
    streamlit_modules = [
        "analytics",
        "crop_knowledge",
        "disease_info",
        "recommendations",
        "app",
    ]

    for mod in utils_modules:
        content = re.sub(
            rf"^[ \t]*(import|from)\s+{mod}\b",
            rf"\1 utils.{mod}",
            content,
            flags=re.MULTILINE,
        )

    for mod in services_modules:
        content = re.sub(
            rf"^[ \t]*(import|from)\s+{mod}\b",
            rf"\1 services.{mod}",
            content,
            flags=re.MULTILINE,
        )

    # For files outside of streamlit folder importing streamlit modules (like tests)
    # Also for files inside streamlit folder trying to import other streamlit modules (since streamlit runs from root now, wait! Streamlit runs from root now?)
    # Yes, we will run `streamlit run streamlit/app.py`. Streamlit adds `streamlit/` to sys.path[0].
    # BUT, to be safe and consistent with the python package system, we should use absolute imports if we set PYTHONPATH=/app.
    # We will update docker-compose and Dockerfile to use `PYTHONPATH=/app`.
    # Therefore, absolute imports like `from streamlit.analytics import ...` are best inside tests.
    if not ("streamlit" in file_path.replace("\\", "/")):
        for mod in streamlit_modules:
            content = re.sub(
                rf"^[ \t]*(import|from)\s+{mod}\b",
                rf"\1 streamlit.{mod}",
                content,
                flags=re.MULTILINE,
            )

    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated imports in {file_path}")


def main():
    for root, dirs, files in os.walk("."):
        if (
            ".git" in root
            or "venv" in root
            or "__pycache__" in root
            or ".pytest_cache" in root
        ):
            continue
        for file in files:
            if file.endswith(".py"):
                update_imports(os.path.join(root, file))


if __name__ == "__main__":
    main()
