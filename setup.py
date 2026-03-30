from qiskit_ibm_runtime import QiskitRuntimeService
import os
from pathlib import Path


def load_dotenv_file(dotenv_path: str = ".env") -> None:
    env_file = Path(dotenv_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        if line.startswith("export "):
            line = line[len("export ") :]

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)


def main():
    load_dotenv_file()

    api_key = os.getenv("QISKIT_API_KEY")
    crn = os.getenv("CRN")
    if not api_key or not crn:
        raise RuntimeError("Missing QISKIT_API_KEY or CRN. Check your .env file.")

    QiskitRuntimeService.save_account(
        token=api_key,
        instance=crn,
    )

if __name__ == "__main__":
    main()