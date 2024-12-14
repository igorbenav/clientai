import os
import subprocess

from dotenv import load_dotenv


def build_docs():
    load_dotenv()

    if not os.getenv("GOOGLE_ANALYTICS_KEY"):
        raise ValueError(
            "GOOGLE_ANALYTICS_KEY environment variable is not set"
        )

    subprocess.run(["mkdocs", "build"], check=True)


if __name__ == "__main__":
    build_docs()
