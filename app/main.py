from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .core.startup import create_app

import os
print(f"CWD: {os.getcwd()}")
loaded = load_dotenv()
print(f"load_dotenv loaded: {loaded}")
print(f"ADMIN_TOKEN from os: {os.getenv('ADMIN_TOKEN')}")

app: FastAPI = create_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

