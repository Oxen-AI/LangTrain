import argparse
from mako.template import Template

from langtrain.models.llms.factory import get_model


def read_prompt(prompt_file: str):
    with open(prompt_file, "r") as f:
        return f.read()


def create_prompt(base_prompt: str, prompt: str):
    return f"{base_prompt}{prompt}"


def main():
    parser = argparse.ArgumentParser(
        prog="Get predictions from an LLM given a file of prompts"
    )

    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-m", "--model", required=True)
    args = parser.parse_args()

    print("Loading model...")
    model = get_model(args.model)

    print("====Input====")
    prompt_template_str = read_prompt(args.prompt)
    prompt_template = Template(prompt_template_str)
    prompt = prompt_template.render(prompt=args.input)

    print(prompt)
    output = model(prompt)
    print("====Output====")
    print(output)


if __name__ == "__main__":
    main()
