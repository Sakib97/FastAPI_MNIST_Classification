# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 22:12:20 2021

"""
from pydantic import BaseModel

class image_path(BaseModel):
    path: str
