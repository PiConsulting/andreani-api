import ml_engine
import click

@click.command()
@click.argument('video_path', type=click.Path(exists=True))
def run(video_path):
    print(ml_engine.run_inference(video_path, debug=True))


if __name__ == "__main__":
    run()