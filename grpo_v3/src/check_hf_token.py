import os


def main() -> None:
    token = os.getenv("HF_TOKEN")

    if token:
        print("HF_TOKEN is set.")
        print("Length:", len(token))
    else:
        print("HF_TOKEN is NOT set.")


if __name__ == "__main__":
    main()

