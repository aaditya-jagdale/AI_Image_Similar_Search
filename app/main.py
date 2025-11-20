from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.debs import router
from pathlib import Path
import logging
from contextlib import asynccontextmanager

#API App
app = FastAPI(title="AI Image Similar Search", version="1.0.0")

#Middleware for security
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["POST", "GET"],
    allow_headers = ["*"],)

#Prefix '/api' endpoint
app.include_router(router=router, prefix="/api")