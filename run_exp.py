import click
import os
from sklearn.metrics import classification_report
from math import ceil
from src import VoiceVer, count_metrics


def test_on_dir(
    app,
    test_dir,
    batch_size=48
):
    name = test_dir.split("/")[-1]
    print("Testing on", name, "directory...")
    X_test, y_test = [], []
    app_classes = app.classifier.classes

    for cls in os.listdir(test_dir):
        for img in os.listdir(os.path.join(test_dir, cls)):
            if cls in app_classes:
                y_test.append(int(cls))
            else:
                y_test.append(-1) # unknown user
            X_test.append(os.path.join(test_dir, cls, img))
    y_pred = []

    for i in range(ceil(len(X_test) / batch_size)):
        batch = X_test[
            i * batch_size:min((i + 1) * batch_size, len(X_test))
        ]
        pred_y = app.identify(batch)
        y_pred.extend(pred_y)

    report = classification_report(y_test, y_pred)
    if report:
        print(report)
        with open(f"reports/report_{name}.txt", "w") as f:
            f.write(report)

def run_exp(
    train_dir,
    mod_dirs,
    config,
    only_clf=False
):
    app = VoiceVer(
        **config
    )
    app.train(train_dir)
    for adv_dir in os.listdir(mod_dirs):
        test_dir = os.path.join(mod_dirs, adv_dir, "test_known")
        test_dir_unknown = os.path.join(mod_dirs, adv_dir, "test_unkown")
        dev_dir = os.path.join(mod_dirs, adv_dir, "dev_known")
        dev_dir_unknown = os.path.join(mod_dirs, adv_dir, "dev_unknown")
        if not only_clf:
            count_metrics(
                app,
                test_dir,
                test_dir_unknown,
                dev_dir,
                dev_dir_unknown,
                batch_size=48
            )
        else:
            test_on_dir(
                app,
                os.path.join(mod_dirs, adv_dir)
            )


@click.command()
@click.option("--train_dir", "-t", required=True)
@click.option("--mod_dirs", "-m", required=True)
@click.option("--cls", "-c", default="DistanceClassifier")
@click.option("--only_clf", "-o", is_flag=True)
def main(
    train_dir,
    mod_dirs,
    cls,
    only_clf=False
):
    config = {
        "backbone": "wespeaker",
        "classifier": cls,
        "decision_th": 0.5,
    }
    run_exp(
        train_dir,
        mod_dirs,
        config,
        only_clf
    )


main()
