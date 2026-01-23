from typing import Tuple
import numpy as np
import smplx
import torch
import os
import trimesh
import pyrender
from PIL import Image
from pyquaternion import Quaternion
from natsort import natsorted
import glob
from utils.smplx_util import SMPLX_Util
from sklearn.neighbors import KDTree
import cv2 as cv

from tools.objectmodel import ObjectModel
from tools.meshviewer import Mesh, MeshViewer, points2sphere, colors
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import makepath
from tools.utils import to_cpu


sequence = "YOUR_HOI_GENERATION.npz"
output_dir = "BODY_SAVE_PATH"
objdir = "HUMAN_SAVE_PATH"
os.makedirs(output_dir)
os.makedirs(objdir)
smplx_folder = './smplv11/models/'
seq_data = parse_npz(sequence)
T = seq_data.n_frames

body_model = smplx.create(smplx_folder, model_type='smplx',
        gender='neutral', ext='npz',
        # num_betas=num_betas,
        use_pca=True,
        num_pca_comps=24,
        create_global_orient=True,
        create_body_pose=True,
        # create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=T,
    )


sbj_mesh = os.path.join(seq_data.body.vtemp)
sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

sbj_m = smplx.create(model_path=smplx_folder,
                        model_type='smplx',
                        num_pca_comps=24,
                        v_template=sbj_vtemp,
                        batch_size=T)
sbj_parms = params2torch(seq_data.body.params)

verts_sbj = (sbj_m(**sbj_parms).vertices).detach().cpu().numpy()

obj_mesh = os.path.join(seq_data.object.object_mesh)
obj_mesh = Mesh(filename=obj_mesh)
obj_vtemp = np.array(obj_mesh.vertices)
obj_m = ObjectModel(v_template=obj_vtemp,
                    batch_size=T)
obj_parms = params2torch(seq_data.object.params)
#obj_parms = params2torch(objpose)
verts_obj = (obj_m(**obj_parms).vertices).detach().cpu().numpy()


for i in range(0,len(verts_sbj),3):
        # print("###")
        # print(i)
        ii = i//3
        body = trimesh.Trimesh(verts_sbj[i], body_model.faces, process=False)
        object = trimesh.Trimesh(verts_obj[i], obj_mesh.faces, process=False)
        body_filename = os.path.join(output_dir, f"body_frame_{ii:04d}.obj")
        body.export(body_filename)

        obj_filename = os.path.join(objdir, f"object_frame_{ii:04d}.obj")
        object.export(obj_filename)


