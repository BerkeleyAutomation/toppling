import numpy as np

from autolab_core import RigidTransform
from dexnet.core import StablePose

up = np.array([0,0,1])

def normalize(vec, axis=None):
    """
    Returns a normalied version of the passed in array

    Parameters
    ----------
    vec : nx3, :obj:`numpy.ndarray` or 3xn :obj:`numpy.ndarray`
    axis : int
        if vec is nx3, axis describes which axis to normalize over

    Returns
    -------
    :obj:`numpy.ndarray` of shape vec
        normalized version of vec
    """
    return vec / np.linalg.norm(vec) if axis == None else vec / np.linalg.norm(vec, axis=axis).reshape((-1,1))

def stable_pose(R):
    """
    Returns a stable pose object from RigidTransform

    Parameters
    ----------
    R : :obj:`autolab_core.RigidTransform`

    Returns
    -------
    :obj:`StablePose`
    """
    if isinstance(R, RigidTransform):
        R = R.matrix
    return StablePose(0, R, eq_thresh=.03, to_frame='world')

def pose_angle(R1, R2):
    """
    Returns difference between two stable poses

    Parameters
    ----------
    R1 : :obj:`autolab_core.RigidTransform`
    R2 : :obj:`autolab_core.RigidTransform`

    Returns
    -------
    int
    """
    z1 = R1.inverse().matrix[:,2]
    z2 = R2.inverse().matrix[:,2]
    return np.arccos(np.clip(z1.dot(z2), 0, 1))

def pose_diff(R1, R2):
    """
    Returns difference between two stable poses

    Parameters
    ----------
    R1 : :obj:`autolab_core.RigidTransform`
    R2 : :obj:`autolab_core.RigidTransform`

    Returns
    -------
    int
    """
    z1 = R1.inverse().matrix[:,2]
    z2 = R2.inverse().matrix[:,2]
    # print z1, z2
    # print np.array(R1.inverse().euler_angles)*180/np.pi
    # print np.array(R2.inverse().euler_angles)*180/np.pi
    return np.linalg.norm(z1 - z2)

def is_equivalent_pose(R1, R2):
    """
    Returns whether the two RigidTransforms are equivalent stable poses

    Parameters
    ----------
    R1 : :obj:`autolab_core.RigidTransform`
    R2 : :obj:`autolab_core.RigidTransform`

    Returns
    -------
    bool
    """
    diff = pose_diff(R1, R2)
    # return -.1 < diff and diff < .1
    return -.2 < diff and diff < .2

def camera_pose():
    CAMERA_ROT = np.array([[ 0,-1, 0],
                           [-1, 0, 0],
                           [ 0, 0,-1]])
    theta = np.pi/3
    c = np.cos(theta)
    s = np.sin(theta)
    CAMERA_ROT = np.array([[c,0,-s],
                           [0,1,0],
                           [s,0,c]]).dot(CAMERA_ROT)
    CAMERA_TRANS = np.array([-.25,-.25,.35])
    CAMERA_TRANS = 1.5*np.array([-.4,0,.3])
    return RigidTransform(CAMERA_ROT, CAMERA_TRANS, from_frame='camera', to_frame='world')