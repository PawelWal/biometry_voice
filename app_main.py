import logging

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.logserver import configure_logger
from app.routers import biom_router
from app.router_utils import voicever


configure_logger()
log = logging.getLogger(__name__)
log.setLevel("INFO")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(biom_router.router)


@app.get("/", include_in_schema=False)
async def root():
    """Endpoint for checking if api is alive."""
    return {"message": "Alive"}


@app.on_event("startup")
def train_voicever():
    """Train voicever on startup."""
    log.info("Training voicever.")
    voicever.train(
        "/mnt/data/bwalkow/vox_new_data/conv_data_joined/train"
    )
    log.info("Voicever training complete.")
