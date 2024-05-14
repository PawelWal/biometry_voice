"""Rest "/biom" router."""
from fastapi import APIRouter, Body, HTTPException
import os

from ..type_declarations import BiomUser, BiomVerify, BiomIdentify
from ..router_utils import voicever


router = APIRouter(
    prefix="/biom"
)

WAITING_MSG = "System is training, please wait."


@router.get("/")
async def root_tasks():
    """Endpoint for checking if task is alive."""
    return {"message": "Tasks alive"}


def check_file_path(file_path):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return None

def unpack_ident(resp):
    classes, probs = resp[0], resp[1]
    print(type(classes), type(probs))
    return [{"class": cls, "prob": prob.tolist()} for cls, prob in zip(classes, probs)]


@router.post("/add_user/",
             tags=["tasks"],
             summary="Add new user to the system")
def add_user(
        user: BiomUser = Body(
            example={
                "user_dir": "/mnt/data/bwalkow/vox_new_data/conv_data_joined/test_unknown/11202"
            }),
):
    """Adds new user to the system."""
    check_file_path(user.user_dir)
    if not voicever.is_training:
        resp = voicever.add_user(user.user_dir)
        return {"new_class": resp}
    else:
        return WAITING_MSG


@router.post("/verify/",
                tags=["tasks"],
                summary="Verify user")
def verify_user(
        user: BiomVerify = Body(
            example={
                "user_file": "/mnt/data/bwalkow/vox_new_data/conv_data_joined/test_known/11249/8uOwUEqSlRM_00003.wav",
                "user_cls": 200
            }),
):
    """Verifies user."""
    check_file_path(user.user_file)
    if not voicever.is_training:
        if str(user.user_cls) in set(voicever.classes):
            resp = voicever.verify([user.user_file], [user.user_cls])
            if isinstance(resp, list):
                return resp[0]
            return resp.tolist()
        else:
            return f"Class {user.user_cls} not registered yet."
    else:
        return WAITING_MSG

@router.post("/identify/",
                tags=["tasks"],
                summary="Identify user")
def identify_user(
        user: BiomIdentify = Body(
            example={
                "user_file": "/mnt/data/bwalkow/vox_new_data/conv_data_joined/test_known/11249/8uOwUEqSlRM_00003.wav"
            }),
):
    """Identifies user."""
    check_file_path(user.user_file)
    if not voicever.is_training:
        resp = voicever.identify([user.user_file])
        if isinstance(resp, list):
            return unpack_ident(resp[0])
        return unpack_ident(resp)
    else:
        return WAITING_MSG


@router.post("/get_files/",
                tags=["tasks"],
                summary="Get images")
def get_files(
        user: BiomUser = Body(
            example={
                "user_dir": "/mnt/data/bwalkow/vox_new_data/conv_data_joined/test_known/11249"
            }),
):
    """Get images."""
    check_file_path(user.user_dir)
    return os.listdir(user.user_dir)


@router.get("/get_classes/",
                tags=["tasks"],
                summary="Get classes")
def get_classes():
    """Get classes."""
    return sorted(list(set(voicever.classes)))


@router.get("/get_num_classes/",
                tags=["tasks"],
                summary="Get classes")
def get_num_classes():
    """Get classes."""
    return len(set(voicever.classes))
