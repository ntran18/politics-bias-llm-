import argparse

from gemini_runner import GeminiBiasRunner
from ollama_runner import OllamaBiasRunner
from openai_runner import OpenAIBatchBiasRunner

from prompt_generation.constants import Constants


def infer_provider(model_name: str) -> str:
    normalized = model_name.lower()
    if normalized.startswith("gpt-") or normalized.startswith("openai/"):
        return "openai"
    if normalized.startswith("gemini-") or normalized.startswith("google/"):
        return "gemini"
    return "ollama"


def build_runner(args):
    provider = args.provider
    if provider == "auto":
        provider = infer_provider(args.model)

    common_kwargs = {
        "model_name": args.model,
        "output_dir": args.output_dir,
        "version": args.version,
        "workers": args.workers,
        "checkpoint_size": args.checkpoint_size,
        "temperature": args.model_temperature,
        "context_length": args.context_length,
    }

    if provider == "openai":
        if args.openai_mode != "batch":
            raise ValueError("Only OpenAI batch mode is currently implemented")

        openai_kwargs = {key: value for key, value in common_kwargs.items() if key != "workers"}
        return OpenAIBatchBiasRunner(
            batch_poll_interval=args.batch_poll_interval,
            **openai_kwargs,
        )

    if provider == "gemini":
        return GeminiBiasRunner(**common_kwargs)

    return OllamaBiasRunner(ollama_port=args.ollama_server_port, **common_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch LLM inference over generated prompts across Ollama/OpenAI/Gemini."
    )

    file_type_choices = ["all"] + list(Constants.PROMPT_FILE_MAP.keys())

    parser.add_argument("--file-type", type=str, default="all", choices=file_type_choices)
    parser.add_argument("--model", type=str, default=Constants.MODEL_NAME)
    parser.add_argument("--output-dir", type=str, default=Constants.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--version", type=str, default=Constants.VERSION)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-size", type=int, default=100)
    parser.add_argument("--model-temperature", type=float, default=0.5)
    parser.add_argument("--prompt-dir", type=str, default=Constants.DEFAULT_PROMPT_DIR)
    parser.add_argument("--ollama-server-port", type=int, default=11434)
    parser.add_argument("--context-length", type=int, default=2048)

    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "ollama", "openai", "gemini"],
        help="Provider override. Default uses model name inference.",
    )
    parser.add_argument(
        "--openai-mode",
        type=str,
        default="batch",
        choices=["batch"],
        help="OpenAI execution mode. Batch is cost-optimized and async up to 24h.",
    )
    parser.add_argument(
        "--batch-poll-interval",
        type=int,
        default=30,
        help="Seconds between OpenAI batch status polls.",
    )

    args = parser.parse_args()

    runner = build_runner(args)
    runner.run_experiment(file_type=args.file_type, prompt_dir=args.prompt_dir)


if __name__ == "__main__":
    main()
