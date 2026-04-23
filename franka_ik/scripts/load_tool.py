import mujoco 
import numpy as np 
from mujoco.viewer import launch 
from pathlib import Path

XML_PATH = "tool.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

viewer = launch(model, data)

while viewer and viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.render()