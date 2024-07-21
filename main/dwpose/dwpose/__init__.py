import os
from DM.dwpose import util
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
import torch.nn as nn

def draw_pose(pose, H, W):
    faces = pose['faces']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = util.draw_facepose(canvas, faces)
    return canvas

class DWposeDetector(nn.Module):
    def __call__(self, oriImg):
        oriImg=oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            return draw_pose(pose, H, W)


