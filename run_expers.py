import tomllib

from rich.console import Console

from train_exper import train

config_dir = "./expers.toml"


def main():
    console = Console()

    print("Loading the config...")
    with open(config_dir, "rb") as file:
        config = tomllib.load(file)

    print("Running the experiments...")
    for key in config:
        console.rule(f"Running experiment: [bold yellow]{key}")
        train({"exper_name": key, **config[key]})


if __name__ == "__main__":
    main()
