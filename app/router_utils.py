from src import VoiceVer

def start_voicever(
    backbone = "wespeaker",
    classifier = "DistanceClassifier",
    decision_th=0.65,
    device_nbr=1,
):
    """Starts face verification system."""
    app = VoiceVer(
            backbone=backbone,
            classifier=classifier,
            decision_th=decision_th,
            device_nbr=device_nbr
        )
    return app


voicever = start_voicever()
