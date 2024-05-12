from src import VoiceVer

def start_facever(
    backbone = "wespeaker",
    classifier = "DistanceClassifier",
    decision_th=0.65,
    device_nbr=1,
):
    """Starts face verification system."""
    app = VoiceVer(
            backbone,
            classifier,
            decision_th,
            device_nbr
        )
    return app


voicever = start_facever()
