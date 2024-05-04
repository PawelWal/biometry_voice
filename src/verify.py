from src import VoiceVer
import click
import os
from sklearn.metrics import classification_report
from math import ceil
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm


def count_metrics(
    app,
    test_dir,
    test_dir_unknown,
    dev_dir,
    dev_dir_unknown,
    batch_size=24,
):
    y_proba_unknown = []
    X_test, X_test_unknown, y_test, y_test_unknown = [], [], [], []
    name = test_dir.split("/")[-2]
    classifier = app.classifier_name
    name = name if test_dir_unknown is not None else f"{name}_unknown_excluded"
    print(f"Testing {name} directory...")
    for cls in tqdm(os.listdir(test_dir), desc="test known"):
        for img in os.listdir(os.path.join(test_dir, cls)):
            y_test.append(int(cls))
            X_test.append(os.path.join(test_dir, cls, img))

    if test_dir_unknown is not None:
        for cls in tqdm(os.listdir(test_dir_unknown), desc="test unknown"   ):
            for img in os.listdir(os.path.join(test_dir_unknown, cls)):
                X_test_unknown.append(os.path.join(test_dir_unknown, cls, img))
                y_test_unknown.append(-1)

        y_pred_unknown, y_proba_unknown = [], []
        for i in range(ceil(len(X_test_unknown) / batch_size)):
            batch = X_test_unknown[
                i * batch_size:min((i + 1) * batch_size, len(X_test_unknown))
            ]
            pred_y, proba = app.identify(batch)
            y_proba_unknown.extend(proba)
            y_pred_unknown.extend(pred_y)

    y_pred = []
    y_proba = []
    for i in range(ceil(len(X_test) / batch_size)):
        batch = X_test[
            i * batch_size:min((i + 1) * batch_size, len(X_test))
        ]
        pred_y, proba = app.identify(batch)
        y_pred.extend(pred_y)
        y_proba.extend(proba)

    # far calculation for impostors & frr calculation for genuine
    right_indexes = [i for i, (x, y) in enumerate(zip(y_test, y_pred)) if x == y]
    miscls = [i for i, (x, y) in enumerate(zip(y_test, y_pred)) if x != y]
    far_mis = []
    far_unknown = []
    frr = []
    threshold = []
    for cur_threshold in range(100):
        num_far_unk, num_far_mis = 0, 0
        num_frr = 0
        if test_dir_unknown is not None:
            for prob in y_proba_unknown:
                if prob * 100 > cur_threshold:
                    num_far_unk += 1

        for idx in miscls:
            if y_proba[idx] * 100 > cur_threshold:
                num_far_mis += 1

        for idx in right_indexes:
            if y_proba[idx] * 100 < cur_threshold:
                num_frr += 1
        # far.append(num_far / len(y_proba_unknown))
        far_mis.append(num_far_mis)
        if test_dir_unknown is not None:
            far_unknown.append(num_far_unk)
        frr.append(num_frr / len(y_test))
        threshold.append(cur_threshold / 100)

    if len(far_unknown) > 0:
        far = np.array(far_mis) + np.array(far_unknown)
        far = far / (len(y_proba_unknown) + len(miscls))
        far = far.tolist()
    else:
        far = np.array(far_mis) / len(miscls)
        far = far.tolist()

    fig, ax = plt.subplots()
    ax.plot(threshold, far, 'r--', label='FAR')
    ax.plot(threshold, frr, 'g--', label='FRR')
    plt.xlabel('Threshold [%]')
    plt.ylabel('Percentage of data [%]')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.savefig(f"metrics/metrics_{classifier}_{name}.png")
    plt.close()

    cross_point = 0
    eer = 0
    for i, (x, y) in enumerate(zip(far, frr)):
        if x <= y:
            cross_point = i
            eer = (x + y)/2
            break
    res_dict = {
        "name": f"{classifier}_{name}",
        "far": far,
        "frr": frr,
        "cross_point": cross_point,
        "eer": eer
    }
    with open(f"metrics/metrics.jsonl", "a") as f:
        f.write(json.dumps(res_dict) + "\n")

    X_test, y_test = [], []
    for cls in os.listdir(dev_dir):
        for img in os.listdir(os.path.join(dev_dir, cls)):
            y_test.append(int(cls))
            X_test.append(os.path.join(dev_dir, cls, img))

    if dev_dir_unknown is not None:
        for cls in os.listdir(dev_dir_unknown):
            for img in os.listdir(os.path.join(dev_dir_unknown, cls)):
                X_test.append(os.path.join(dev_dir_unknown, cls, img))
                y_test.append(-1)

    y_pred = []

    app.decision_th = cross_point / 100
    for i in range(ceil(len(X_test) / batch_size)):
        batch = X_test[
            i * batch_size:min((i + 1) * batch_size, len(X_test))
        ]
        pred_y, proba = app.identify(batch)
        y_pred.extend(pred_y)

    report = classification_report(y_test, y_pred)
    with open(f"metrics/report_{classifier}_{name}.txt", "w") as f:
        f.write(report)
