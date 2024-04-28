from src import VoiceVer, count_metrics
import click
import os
from sklearn.metrics import classification_report
from math import ceil
import matplotlib.pyplot as plt
import json


@click.command()
@click.option("--data_dir", "-t", required=True)
@click.option("--backbone", "-b", default="wespeaker")
@click.option("--classifier", "-c", default="DistanceClassifier")
@click.option("--decision_th", "-d", default=0.5)
@click.option("--bs", default=48)
@click.option("--exclude_unknown", "-eu", is_flag=True)
def main(
    data_dir,
    backbone,
    classifier,
    decision_th,
    bs,
    exclude_unknown=False
):
    test_dir = os.path.join(data_dir, "test_known")
    test_dir_unknown = os.path.join(data_dir, "test_unkown")
    dev_dir = os.path.join(data_dir, "dev_known")
    dev_dir_unknown = os.path.join(data_dir, "dev_unknown")
    train_dir = os.path.join(data_dir, "train")
    print("Name", test_dir.split("/")[-2])
    app = VoiceVer(
        backbone,
        classifier,
        decision_th
    )
    app.train(train_dir)
    count_metrics(
        app,
        test_dir,
        test_dir_unknown if not exclude_unknown else None,
        dev_dir,
        dev_dir_unknown if not exclude_unknown else None,
        bs
    )


main()
