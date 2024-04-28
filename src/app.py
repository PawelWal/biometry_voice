import os
import numpy as np
from .backbone import build_representation
from importlib import import_module
from time import time
from collections import defaultdict
from time import time
from copy import deepcopy
import wespeaker


class VoiceVer:

    def __init__(
        self,
        backbone = "wespeaker",
        classifier = "SVMClassifier",
        decision_th=0.5,
        device_nbr=1,
    ):
        self.model = wespeaker.load_model('english')
        self.model.set_gpu(1)
        self.model_name = "wespeaker"
        self.backbone = backbone
        self.decision_th = decision_th
        self.classifier_name = classifier
        self.classifier = None
        self.classes = []
        self.is_training = False
        self.X_rep = []
        self.y = []

    def build_representation(self, audio_list, verbose=False):
        return build_representation(
            self.model,
            audio_list,
            method=self.model_name,
            verbose=verbose
        )

    def train(self, train_dir):
        self.is_training = True
        X, y = [], []
        for cls in os.listdir(train_dir):
            for img in os.listdir(os.path.join(train_dir, cls)):
                X.append(os.path.join(train_dir, cls, img))
                y.append(cls)

        assert np.array(X).shape[0] == np.array(y).shape[0] # sanity check
        start = time()
        X_rep = self.build_representation(X, verbose=True)
        print(f"Building representation took {time() - start}")
        self.X_rep = X_rep
        self.y = y
        assert np.array(X_rep).shape[0] == np.array(y).shape[0]
        self.__train(X_rep, y)
        self.is_training = False
        print("Training done")

    def __train(self, X_rep, y):
        clf_class = getattr(
            import_module("src.classifier"),
            self.classifier_name
        )
        if self.classifier_name == "KNNClassifier":
            classes_dict = defaultdict(int)
            for cls in y:
                classes_dict[cls] += 1
            self.classifier = clf_class(
                self.decision_th,
                min(classes_dict.values())
            )
        else:
            self.classifier = clf_class(self.decision_th)
        start = time()
        print(f"Training classifier {self.classifier_name} "
              f"with {len(X_rep)} and {len(set(y))} classes")
        self.classifier.train(X_rep, y)
        self.classes = self.classifier.classes
        print(f"Training done in {time() - start}")

    def add_user(
        self,
        user_dir,
    ):
        X, y = [], []
        try:
            user_cls = int(user_dir.split("/")[-1])
        except ValueError:
            raise ValueError(f"User directory must be a number, found {user_dir} instead")

        for img in os.listdir(user_dir):
            X.append(os.path.join(user_dir, img))
            y.append(user_cls)
        X_rep = self.build_representation(X)
        self.X_rep.extend(X_rep)
        self.y.extend(y)
        self.__train(self.X_rep, self.y) # retrain with new user
        return user_cls

    def verify(
        self,
        user_img,
        user_cls
    ):
        if self.classifier is None:
            raise ValueError("Classifier is not trained yet")
        user_img = [user_img] if not isinstance(user_img, list) else user_img
        user_cls = [user_cls] if not isinstance(user_cls, list) else user_cls
        if len(user_img) != len(user_cls):
            raise ValueError("Number of images and classes must be equal")
        user_rep = self.build_representation(user_img)
        return self.classifier.verify_cls(user_rep, user_cls)

    def identify(
        self,
        user_img
    ):
        """
        Return the class of the user -1 if not found
        """
        if self.classifier is None:
            raise ValueError("Classifier is not trained yet")
        user_img = [user_img] if not isinstance(user_img, list) else user_img
        user_rep = self.build_representation(user_img)
        return self.classifier.predict(user_rep)
