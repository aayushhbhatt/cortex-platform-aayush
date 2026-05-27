from agents.supervisor import invoke_with_trace


def main() -> None:
    """Run Cortex with a small Story 2 RAG demo."""
    print("hello cortex")

    final_state = invoke_with_trace("What is the parental leave policy?")
    print(f"Demo response: {final_state['response']}")


if __name__ == "__main__":
    main()
