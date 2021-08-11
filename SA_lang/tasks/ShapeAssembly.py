# Taken from https://github.com/rkjones4/ShapeAssembly

import torch
import re
import numpy as np
import math
import ast
import sys
import faiss
from copy import deepcopy

"""
  This file contains all of the logic in the ShapeAssembly DSL.

  You can execute a ShapeAssembly program as follows:

  > from ShapeAssembly import ShapeAssembly
  > sa = ShapeAssembly()
  > lines = sa.load_lines({path_to_program})
  > sa.run(lines, {out_file_name})

  The classes in this file are:

  Cuboid -> Part Proxies represented as Cuboids in space
  AttPoint -> Points that live within the local coordinate frames of cuboids -> specify where cuboids should attach
  Program -> Parses lines, locally executes lines by creating Cuboids, AttPoints, and changing their attributes. 
  ShapeAssembly -> Entrypoint to language logic

"""

# Params controlling execution behavior
EPS = .01
SMALL_EPS = 1e-4
COS_DIST_THRESH = 0.9


# Helper function: write mesh to out file
def writeObj(verts, faces, outfile):
    with open(outfile, 'w') as f:
        for a, b, c in verts.tolist():
            f.write(f'v {a} {b} {c}\n')
        for a, b, c in faces.tolist():
            f.write(f"f {a+1} {b+1} {c+1}\n")

def samplePC(cubes, xyz1, xyz2):
    cube_geom = []
    for c in cubes:
        cube_geom.append(torch.cat((
            c['xd'].unsqueeze(0),
            c['yd'].unsqueeze(0),
            c['zd'].unsqueeze(0),
            c['center'],            
            c['xdir'],
            c['ydir']
        )))
        
    scene_geom = torch.stack([c for c in cube_geom])
    ind_to_pc = {}

    xyzs = [xyz1, xyz2]
    
    for i in range(0, scene_geom.shape[0]):

        xyz = xyzs[i]
        
        s_inds = (torch.ones(1,xyz.shape[1]) * i).long()
        
        s_r = torch.cat(
            (
                (scene_geom[s_inds][:, :, 6:9] / (scene_geom[s_inds][:, :, 6:9].norm(dim=2).unsqueeze(2) + 1e-8)).unsqueeze(3),
                (scene_geom[s_inds][:, :, 9:12] / (scene_geom[s_inds][:, :, 9:12].norm(dim=2).unsqueeze(2) + 1e-8)).unsqueeze(3),
                torch.cross(
                    scene_geom[s_inds][:, :, 6:9] / (scene_geom[s_inds][:, :, 6:9].norm(dim=2).unsqueeze(2) + 1e-8),
                    scene_geom[s_inds][:, :, 9:12] / (scene_geom[s_inds][:, :, 9:12].norm(dim=2).unsqueeze(2) + 1e-8)                
                ).unsqueeze(3)
            ), dim = 3)
    
        s_out = ((s_r @ (((xyz - .5) * scene_geom[s_inds][:, :, :3]).unsqueeze(-1))).squeeze() + scene_geom[s_inds][:, :, 3:6]).squeeze()
        ind_to_pc[i] = s_out

    res = {}
    for key in ind_to_pc:
        index = faiss.IndexFlatL2(3)            
        index.add(
            np.ascontiguousarray(ind_to_pc[key].cpu().numpy())
        )
        res[key] = (ind_to_pc[key], index)
        
    return res
            
# Helper function: given angle + normal compute a rotation matrix that will accomplish the operation
def getRotMatrix(angle, normal):
    s = torch.sin(angle)
    c = torch.cos(angle)

    nx = normal[0]
    ny = normal[1]
    nz = normal[2]
        
    rotmat = torch.stack((
        torch.stack((c + (1 - c) * nx * nx, (1 - c) * nx * ny - s * nz, (1 - c) * nx * nz + s * ny)),
        torch.stack(((1 - c) * nx * ny + s * nz, c + (1 - c) * ny * ny, (1 - c) * ny * nz - s * nx)),
        torch.stack(((1 - c) * nx * nz - s * ny, (1 - c) * ny * nz + s * nx, c + (1 - c) * nz * nz))
    ))
    return rotmat


# Helper function: Find a minimum rotation from the current direction to the target direction
def findMinRotation(cur, target):
        
    assert(cur.norm() != 0)
    assert(target.norm() != 0)
        
    ncur = cur / cur.norm() 
    ntarget = target / target.norm()
        
    normal = torch.cross(ncur, ntarget)

    # co-linear
    if normal.norm() == 0:
        r_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0, 1.0, 0.0]])
        r_y = torch.tensor([[0.0, 0, 1.0], [0.0, 1.0, 0.0], [ -1.0, 0.0, 0.0]])
        r_z = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            
        if torch.dot(r_x @ ncur, ncur) != 0:
            cnormal = r_x @ ncur
        elif torch.dot(r_y @ ncur, ncur) != 0:
            cnormal = r_y @ cur
        elif torch.dot(r_z @ ncur, ncur) != 0:
            cnormal = r_z @ ncur

        assert(cnormal.norm() != 0)
        nnormal = cnormal / cnormal.norm()
        angle = torch.tensor(math.pi)

    else:
        
        nnormal = normal / normal.norm()
        angle = torch.acos(torch.dot(ncur, ntarget))
        if angle == 0 or torch.isnan(angle).any():
            return torch.eye(3)

    return getRotMatrix(angle, nnormal)



class Cuboid():
    """
    Cuboids are the base (and only) objects of a ShapeAssembly program. Dims are their dimensions, pos is the center of the cuboid, rfnorm (right face), tfnorm (top face) and ffnorm (front face) specify the orientation of the cuboid. The bounding volume is just a non-visible cuboid. Cuboids marked with the aligned flag behavior differently under attachment operations. 
    """
    def __init__(self, name, aligned = False, vis = True):
        
        # The default cube is unit, axis-aligned, centered at the origin
        self.dims =  torch.tensor([1.0,1.0,1.0])
        self.pos = torch.tensor([0.0,0.0,0.0])
        self.rfnorm = torch.tensor([1.0,0.0,0.0])
        self.tfnorm = torch.tensor([0.0,1.0,0.0])
        self.ffnorm = torch.tensor([0.0,0.0,1.0])
        # Keep track of all attachment obligations this cube has
        self.attachments = []
        self.move_atts = []
        # The bbox is not visible, but is still a cuboid, otherwise this should be True
        self.is_visible = vis
        self.name = name
        self.parent = None
        self.parent_axis = None
        self.aligned = aligned

    # Rotate the cuboid by the rotation matrix
    def rotateCuboid(self, rotation):
        self.rfnorm = rotation @ self.rfnorm
        self.tfnorm = rotation @ self.tfnorm
        self.ffnorm = rotation @ self.ffnorm

    def flipCuboid(self, a_ind):
        transform = torch.ones(3)
        transform[a_ind] *= -1 
        self.pos = transform * self.pos
        self.rfnorm = -1 * (transform * self.rfnorm)
        self.tfnorm = -1 * (transform * self.tfnorm)
        self.ffnorm = -1 * (transform * self.ffnorm)
        
    # Get the corners of the cuboid
    def getCorners(self):
        xd = self.dims[0] / 2
        yd = self.dims[1] / 2
        zd = self.dims[2] / 2

        corners = torch.stack((
            (self.rfnorm * xd) + (self.tfnorm * yd) + (self.ffnorm * zd),
            (self.rfnorm * xd) + (self.tfnorm * yd) + (self.ffnorm * -1 * zd),
            (self.rfnorm * xd) + (self.tfnorm * -1 * yd) + (self.ffnorm * zd),
            (self.rfnorm * xd) + (self.tfnorm * -1 * yd) + (self.ffnorm * -1 * zd),
            (self.rfnorm * -1 * xd) + (self.tfnorm * yd) + (self.ffnorm * zd),
            (self.rfnorm * -1 * xd) + (self.tfnorm * yd) + (self.ffnorm * -1 * zd),
            (self.rfnorm * -1 * xd) + (self.tfnorm * -1 * yd) + (self.ffnorm * zd),
            (self.rfnorm * -1 * xd) + (self.tfnorm * -1 * yd) + (self.ffnorm * -1 * zd),
        ))
        return corners + self.pos

    # Get the global point specified by relative coordinates x,y,z 
    def getPos(self, x, y, z):
        
        pt = torch.stack((x, y, z))
    
        r = torch.stack((
            self.rfnorm,
            self.tfnorm,
            self.ffnorm
        )).T

        t_dims = torch.stack((self.dims[0], self.dims[1], self.dims[2]))
        
        return (r @ ((pt - .5) * t_dims)) + self.pos

    # Get the relative position of global poiunt gpt
    def getRelPos(self, gpt, normalize = False):
        O = self.getPos(
            torch.tensor(0.),
            torch.tensor(0.),
            torch.tensor(0.)
        )
        A = torch.stack([
            self.dims[0].clone() * self.rfnorm.clone(),
            self.dims[1].clone() * self.tfnorm.clone(),
            self.dims[2].clone() * self.ffnorm.clone()
        ]).T

        B = gpt - O
        p = A.inverse() @ B

        if normalize:
            return torch.clamp(p, 0.0, 1.0)
        
        return p                
    
    # Make the cuboid bigger by a multiplied factor of scale (either dim 3 or dim 1)
    def scaleCuboid(self, scale):
        self.dims *= scale

    # Make the cuboid bigger by an added factor of scale to a specific dimension
    def increaseDim(self, dim, inc):
        dim_to_scale = {            
            "height": torch.tensor([0.0, 1.0, 0.0]),
            "width": torch.tensor([0.0, 0.0, 1.0]),
            "length": torch.tensor([1.0, 0.0, 0.0])
        }
        s = dim_to_scale[dim] * inc
        self.dims += s
        
    # Move the center of the cuboid by the translation vector
    def translateCuboid(self, translation):
        self.pos += translation

    # Used to convert cuboid into triangles on its faces
    def getTriFaces(self):
        return [
            [0, 2, 1],
            [1, 2, 3],
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 5],
            [3, 7, 5],
            [4, 5, 7],
            [4, 7, 6],
            [1, 5, 4],
            [0, 1, 4],
            [2, 6, 7],
            [2, 7, 3]
        ]

    # Get the triangulation of the cuboid corners, for visualization + sampling
    def getTris(self):
        if self.is_visible:
            verts = self.getCorners()
            faces = torch.tensor(self.getTriFaces(), dtype=torch.long)
            return verts, faces
        return None, None

    # Return any attachments that are on this cuboid
    def getAttachments(self):
        return self.attachments
    
    # Return the cuboid's parameterization
    def getParams(self):
        return torch.cat((
            self.dims, self.pos, self.rfnorm, self.tfnorm, self.ffnorm
        )) 

class AttPoint():
    """ 
    Attachment Points live with the local coordinate frame [0, 1]^3 of a cuboid. They are used to connect cuboids together.
    """
    def __init__(self, cuboid, x, y, z):
        self.cuboid = cuboid
        self.x = x
        self.y = y
        self.z = z

    # To get the global position, all we need is the cuboid+face info, and the relative uv pos
    def getPos(self):
        return self.cuboid.getPos(self.x, self.y, self.z)
    
    # If we scale the height of the cuboid, what is the rate of change of this AP
    def getChangeVectorHeight(self):
        norm = self.cuboid.tfnorm
        return (self.y - .5) * norm

    # If we scale the length of the cuboid, what is the rate of change of this AP
    def getChangeVectorLength(self):
        norm = self.cuboid.rfnorm
        return (self.x - .5) * norm
        
    # If we scale the width of the cuboid, what is the rate of change of this AP
    def getChangeVectorWidth(self):
        norm = self.cuboid.ffnorm
        return (self.z - .5) * norm
        
    # get rate of change of this AP when we change the specified dimension
    def getChangeVector(self, dim):
        dim_to_sf = {
            'height': self.getChangeVectorHeight,
            'length': self.getChangeVectorLength,
            'width': self.getChangeVectorWidth,
        }
        return dim_to_sf[dim]()                

    # If we scale the height of the cuboid, what direction does the AP move with
    def getChangeDirHeight(self):
        if self.y > .5:
            return 'top'
        elif self.y < .5:
            return 'bot'
        else:
            return 'none'

    # If we scale the length of the cuboid, what direction does the AP move with
    def getChangeDirLength(self):
        if self.x > .5:
            return 'right'
        elif self.x < .5:
            return 'left'
        else:
            return 'none'    

    # If we scale the width of the cuboid, what direction does the AP move with
    def getChangeDirWidth(self):        
        if self.z > .5:
            return 'front'
        elif self.z < .5:
            return 'back'
        else:
            return 'none'
    
    def getChangeDir(self, dim):
        dim_to_dir = {
            'height': self.getChangeDirHeight,
            'length': self.getChangeDirLength,
            'width': self.getChangeDirWidth,
        }
        return dim_to_dir[dim]()
    


class Program():
    """
    A program maintains a representation of entire shape, including all of the member cuboids
    and all of the attachment points. The execute function is the entrypoint of text programs.
    """
    def __init__(self, cuboids = {}):
        self.cuboids = self.getBoundBox()
        self.cuboids.update(cuboids)
        self.commands = []
        self.parameters = []
        self.att_points = {}

        self.resource = None
        
        # CONSTANTS For PC Intersection
        DIM = 10
        a = (torch.arange(DIM).float()/(DIM-1))
        b = a.unsqueeze(0).unsqueeze(0).repeat(DIM, DIM, 1)
        c = a.unsqueeze(0).unsqueeze(2).repeat(DIM, 1, DIM)
        d = a.unsqueeze(1).unsqueeze(2).repeat(1, DIM, DIM)
        g = torch.stack((b,c,d), dim=3).view(-1, 3)
        self.s_xyz = g.unsqueeze(0)
        bb_top_inds = (g[:,1] == 1.0).nonzero().squeeze()
        bb_bot_inds = (g[:,1] == 0.0).nonzero().squeeze()
        self.bb_top_xyz = g[bb_top_inds].unsqueeze(0)
        self.bb_bot_xyz = g[bb_bot_inds].unsqueeze(0)
        
    def flip(self, flip_axis):
        if flip_axis == 'X':
            axis = 0
        elif flip_axis == 'Y':
            axis = 1
        elif flip_axis == 'Z':
            axis = 2
        for name, c in self.cuboids.items():
            if name == 'bbox':
                continue
            c.flipCuboid(axis)
            
    # Each program starts off with an invisible bounding box
    def getBoundBox(self):
        bbox = Cuboid("bbox", aligned = True, vis=False)
                
        return {
            "bbox": bbox
        }


    # Get the triangles in the current scene -> first index is bounding box so skipped
    def getShapeGeo(self):
        
        if len(self.cuboids) < 2:
            return None, None
        
        cuboids = list(self.cuboids.values())
        
        verts = torch.tensor([],dtype=torch.float)
        faces = torch.tensor([],dtype=torch.long)
        
        for cube in cuboids[1:]:            
            v, f = cube.getTris()
            if v is not None and f is not None:
                faces =  torch.cat((faces, (f + verts.shape[0])))
                verts = torch.cat((verts, v))

        return verts, faces


    # Make an obj of the current scene
    def render(self, ofile = "output.obj"):        
        verts, faces = self.getShapeGeo()
        writeObj(verts, faces, ofile)

    # Parses a cuboid text line
    def parseCuboid(self, line):
        s = re.split(r'[()]', line)
        name = s[0].split("=")[0].strip()
        dim0 = None
        dim1 = None
        dim2 = None
        aligned = False

        params = s[1].split(',')
        dim0 = torch.tensor(float(params[0]))
        dim1 = torch.tensor(float(params[1]))
        dim2 = torch.tensor(float(params[2]))
        if len(params) == 4:
            aligned = ast.literal_eval(params[3].strip())
        assert isinstance(aligned, bool), 'aligned not a bool'
        return (name, dim0, dim1, dim2, aligned)

    
    # Construct a new cuboid, add it to state
    def executeCuboid(self, parse):
        name = parse[0]

        if name in self.cuboids:
            c = self.cuboids[name]            
            c.dims = torch.stack((parse[1], parse[2], parse[3]))
            
        else:            
            c = Cuboid(
                parse[0],
                aligned = parse[4],
            )
            
            c.scaleCuboid(torch.stack((parse[1], parse[2], parse[3])))

            self.cuboids.update({
                parse[0]: c
            })
            
    # Logic for cuboids with no previous attachment. Finds a translation to satisfy the attachment
    def first_attach(self, ap, gpos):
        cur_pos = ap.getPos()
        diff = gpos - cur_pos
        ap.cuboid.translateCuboid(diff)
        return True
        
    # Logic for unaligned cuboids with one previous attachment. Find a scale and rotation to satisfy the attachment
    def second_attach(self, ap, gpos, prev_att):
        p_ap = prev_att[0]
        p_gpos = prev_att[1]
        
        a = p_gpos
        b = ap.getPos()
        c = gpos

        if (b-c).norm() < SMALL_EPS:
            return True
        
        # Increase dimension to fix distance
        dist = (c-a).norm()
        min_dim = 'height'
        min_sf = 1e8

        for dim in ('height', 'width', 'length'):
            
            nsf = ap.getChangeVector(dim)
            psf = p_ap.getChangeVector(dim)

            if nsf.abs().sum() + psf.abs().sum() < SMALL_EPS:
                continue
                        
            cn = b - a
            dn = nsf - psf
            
            at = (dn**2).sum()
            bt = 2 * (cn*dn).sum()
            ct = (cn**2).sum() - (dist**2)

            # Take the positive solution of the quadratic equation
            sf = ((-1 * bt) + (bt**2 - (4*at*ct) ).sqrt()) / ((2 * at) + 1e-8)            
            if abs(sf) < abs(min_sf) and (bt**2 - (4*at*ct)) > 0:
                min_sf = sf
                min_dim = dim
                
        if min_sf ==  1e8:                        
        
            nsf = ap.getChangeVector('height') + \
                  ap.getChangeVector('length') + \
                  ap.getChangeVector('width')
            
            psf = p_ap.getChangeVector('height') + \
                  p_ap.getChangeVector('length') + \
                  p_ap.getChangeVector('width')
            
            cn = b - a
            dn = nsf - psf
            
            at = (dn**2).sum()
            bt = 2 * (cn*dn).sum()
            ct = (cn**2).sum() - (dist**2)

            # Take the positive solution of the quadratic equation
            sf = ((-1 * bt) + (bt**2 - (4*at*ct) ).sqrt()) / ((2 * at) + 1e-8)

            if not torch.isnan(sf) and (bt**2 - (4*at*ct)) > 0:            
                ap.cuboid.increaseDim('height', sf)
                ap.cuboid.increaseDim('length', sf)
                ap.cuboid.increaseDim('width', sf)                        

        else:
            ap.cuboid.increaseDim(min_dim, min_sf)

        # Reset the position of the cuboid such that the previous attachment is satisfied
        diff = p_gpos - p_ap.getPos()
        ap.cuboid.translateCuboid(diff)
        
        # find rotation to match points

        nb = ap.getPos() - p_gpos
        nc = c - p_gpos
        
        # If we are already in the correct position, don't rotate
        if nb.norm() == 0 or nc.norm() == 0 or (nb-nc).norm() < SMALL_EPS:
            return True

        rot_mat = findMinRotation(nb, nc)
        ap.cuboid.rotateCuboid(rot_mat)
        
        # Reset the position of the cuboid such that the attachments are satisfied
        sdiff = p_gpos - p_ap.getPos()
        
        ap.cuboid.translateCuboid(sdiff)

        return True

    # Returns if all of the previous attachment points on cuboid are co-linear
    def checkColinear(self, prev_atts):
        if len(prev_atts) == 2:
            return True
        else:
            o = prev_atts[0][1]
            l = prev_atts[1][1] - o
            for i in range(2, len(prev_atts)):
                n = prev_atts[i][1] - o
                if torch.cross(n, l).norm() > EPS:
                    return False
            return True

    # For cuboids with 2+ attachments, checks if changing orientation would improve the fit
    def doOrient(self, ap, gpos, prev_atts):        
        p_ap0 = prev_atts[0][0]
        p_gpos0 = prev_atts[0][1]

        p_ap1 = prev_atts[1][0]
        p_gpos1 = prev_atts[1][1]
        
        a0 = p_gpos0
        a1 = p_gpos1
        
        b = ap.getPos() 
        c = gpos

        axis = (a0 - a1) / ((a0 - a1).norm() + 1e-8)
        
        # project c onto the plane defined by b and the axis of rotation
        tc = (axis.dot(b) - axis.dot(c)) / (axis.dot(axis) + 1e-8)
        c_p = c + (tc * axis)
        
        # project a0 onto the plane defined by b and the axis of rotation, this is the origin of rotation
        
        ta = (axis.dot(b) - axis.dot(a0)) / (axis.dot(axis) + 1e-8)
        a_p = a0 + (ta * axis)

        # project center of the cube onto the plane defined by b and the axis of rotation
        center = ap.cuboid.pos
        to = (axis.dot(b) - axis.dot(center)) / (axis.dot(axis) + 1e-8)
        o_p = center + (to * axis)

        fn = (b - a_p)
        
        noc = c_p - a_p
       
        at = fn.dot(fn)
        bt = 2 * (b-a_p).dot(fn)
        ct = (b-a_p).dot((b-a_p)) - noc.dot(noc)

        sf = ((-1 * bt) + (bt**2 - (4*at*ct) ).sqrt()) / ((2 * at) + 1e-8)

        if torch.isnan(sf):
            return False
        
        b_p = b + (sf * fn)
        nob = b_p - a_p
        
        # If we are already in the correct position, don't rotate
        if nob.norm() == 0 or noc.norm() == 0 or (nob-noc).norm() < EPS:
            return True
        
        nb = nob / (nob.norm() + 1e-8)
        nc = noc / (noc.norm() + 1e-8)
        
        angle = torch.acos(torch.dot(nb, nc))
        rot_mat = getRotMatrix(angle, axis)
        
        # make sure axis is oriented correctly
        if ((rot_mat @ nb) - nc).norm() > EPS:
            rot_mat = getRotMatrix(angle, -1*axis)

        ap.cuboid.rotateCuboid(rot_mat)
        diff = a0 - p_ap0.getPos()
        ap.cuboid.translateCuboid(diff)

        
    # For cuboids with 2+ attachments, checks if scaling any dimension would improve the fit
    def maybeScale(self, ap, gpos, prev_atts):
        b = ap.getPos()
        c = gpos
        
        t_dir = c - b
        if t_dir.norm() < EPS:
            return

        # Can only scale a dimension if all of the CV for that dimension are the same
        eq_cv_dims = []
        for dim in ('height', 'width', 'length'):
            cv = prev_atts[0][0].getChangeVector(dim)
            add = True
            for p_ap in prev_atts:
                if (cv - p_ap[0].getChangeVector(dim)).norm() > EPS:
                    add = False
            if add:
                eq_cv_dims.append(dim)
                
        dirs_to_avoid = set()
                
        best_dim = None
        cos_dist = COS_DIST_THRESH
        
        nt_dir = t_dir / (t_dir.norm() + 1e-8)

        # See if scaling any direction would improve the fit (i.e. has a valid solution to quadratic equation)
        
        for dim in eq_cv_dims:
            sf = ap.getChangeVector(dim)
            dr = ap.getChangeDir(dim)
            
            if dr in dirs_to_avoid:
                continue
            
            nsf = sf / (sf.norm() + 1e-8)
            
            if torch.isnan(nsf).any().tolist():
                continue
            
            if nt_dir.dot(nsf) > cos_dist:
                best_dim = dim
                cost_dist = nt_dir.dot(nsf)

        if best_dim == None:
            return False
        
        axis = ap.getChangeVector(best_dim)
        paxis = prev_atts[0][0].getChangeVector(best_dim)

        d = 1.0 / ((axis-paxis).norm() + 1e-8)

        naxis = axis / (axis.norm() + 1e-8)
        
        t = (naxis.dot(b) - naxis.dot(c)) / (naxis.dot(naxis) + 1e-8)
        c_p = c + (t * naxis)
        offset = c_p - b
        c_g = c - offset

        sf = (c_g - b).norm()
        
        ap.cuboid.increaseDim(best_dim, sf*d)
            
        diff = prev_atts[0][1] - prev_atts[0][0].getPos()
        ap.cuboid.translateCuboid(diff)

    # Given cuboids a and b, find the closest pair of points in their local coordinate frames to one another
    def getClosestPoints(self, a, b, xyz1, xyz2, is_bbox):

        if not is_bbox:
        
            a_corners = a.getCorners()
            for i in range(a_corners.shape[0]):
                rp = b.getRelPos(a_corners[i])
                if rp.min() >= 0.0 and rp.max() <= 1.0:
                    return None, None

            b_corners = b.getCorners()
            for i in range(b_corners.shape[0]):
                rp = a.getRelPos(b_corners[i])
                if rp.min() >= 0.0 and rp.max() <= 1.0:
                    return None, None
                    
        ind_to_pc = samplePC(
            [
                {
                    'xd': a.dims[0].detach(),
                    'yd': a.dims[1].detach(),
                    'zd': a.dims[2].detach(),
                    'center': a.pos.detach(),
                    'xdir': a.rfnorm.detach(),
                    'ydir': a.tfnorm.detach(),
                },
                {
                    'xd': b.dims[0].detach(),
                    'yd': b.dims[1].detach(),
                    'zd': b.dims[2].detach(),
                    'center': b.pos.detach(),
                    'xdir': b.rfnorm.detach(),
                    'ydir': b.tfnorm.detach()
                }
            ],
            xyz1,
            xyz2
        )

        a_pc = ind_to_pc[0][0].unsqueeze(0)
        b_pc = ind_to_pc[1][0].unsqueeze(0)
        b_index = ind_to_pc[1][1]
        
        a_query = np.ascontiguousarray(
            a_pc.data.clone().detach().cpu().numpy()
        )

        D, I = b_index.search(a_query[0], 1)
        
        I_var = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64)).squeeze()
        D_var = torch.from_numpy(np.ascontiguousarray(D).astype(np.float)).squeeze()
        
        a_ind = D_var.argmin()
        b_ind = I_var[a_ind]

        a_pt = a_pc[0, a_ind]
        b_pt = b_pc[0, b_ind]

        return a_pt.cpu(), b_pt.cpu()
        
    # For aligned cuboids with a previous attachments,
    # see if increasing any dimension would cause the fit to be improved
    def aligned_attach(self, ap, oap):

        if oap.cuboid.name != 'bbox':
            xyz2 = self.s_xyz
        else:
            if oap.y > 0.5:
                xyz2 = self.bb_top_xyz
            else:
                xyz2 = self.bb_bot_xyz
        
        b, c = self.getClosestPoints(
            ap.cuboid,
            oap.cuboid,
            self.s_xyz,
            xyz2,
            oap.cuboid.name == 'bbox'
        )
        
        if b is None or c is None:
            return
        
        mv = max(
            ap.cuboid.dims.max().item(),
            oap.cuboid.dims.max().item()
        )

        # Do a sampling of 20 in inter
        if (b-c).norm() < mv * .05:
            return

        rp = ap.cuboid.getRelPos(b, True)
        
        n_ap = AttPoint(ap.cuboid, rp[0], rp[1], rp[2])
        o_ap = AttPoint(ap.cuboid, 1-rp[0], 1-rp[1], 1-rp[2])

        ppos = o_ap.getPos()        
        
        D_l = 0
        D_h = 0
        D_w = 0

        l_cv = n_ap.getChangeVector('length')[0]
        if l_cv != 0:
            D_l = (c[0] - b[0]) / (2*l_cv)

        h_cv = n_ap.getChangeVector('height')[1]
        if h_cv != 0:
            D_h = (c[1] - b[1]) / (2*h_cv)
            
        w_cv = n_ap.getChangeVector('width')[2]
        if w_cv != 0:
            D_w = (c[2] - b[2]) / (2*w_cv)

        if D_l > 0:
            ap.cuboid.increaseDim("length", D_l)
        if D_h > 0:
            ap.cuboid.increaseDim("height", D_h)
        if D_w > 0:
            ap.cuboid.increaseDim("width", D_w)

        diff = ppos - o_ap.getPos()
        ap.cuboid.translateCuboid(diff)
                
        
    # Attachment for a non-aligned cuboid with 2+ previous attachments. Only try to orient the cuboid if its previous attachment are
    # all colinear.
    def gen_attach(self, ap, gpos, prev_atts):
        if self.checkColinear(prev_atts): 
            self.doOrient(ap, gpos, prev_atts)

        self.maybeScale(ap, gpos, prev_atts)

        
    # Moves the attach point to the global position
    def attach(self, ap, gpos, oci, oap=None):
        assert ap.cuboid.name != "bbox", 'tried to move the bbox'
        
        if ap.cuboid.aligned:
            self.aligned_cube_attach(ap, gpos, oci, oap)
        else:
            self.free_cube_attach(ap, gpos, oci)

    # Aligned attachment
    def aligned_cube_attach(self, ap, gpos, oci, oap):
        prev_atts = ap.cuboid.getAttachments()
        if len(prev_atts) == 0:
            self.first_attach(ap, gpos)
        else:
            self.aligned_attach(ap, oap)
        
        prev_atts.append((ap, gpos, oci))
        ap.cuboid.move_atts.append((ap, gpos, oci))

    # Non-aligned attachment
    def free_cube_attach(self, ap, gpos, oci):
        prev_atts = ap.cuboid.getAttachments()
                
        if len(prev_atts) == 0:
            self.first_attach(ap, gpos)
        elif len(prev_atts) == 1:
            self.second_attach(ap, gpos, prev_atts[0])
        else:
            self.gen_attach(ap, gpos, prev_atts)

        prev_atts.append((ap, gpos, oci))
        ap.cuboid.move_atts.append((ap, gpos, oci))
        
    # Parses an attach line
    def parseAttach(self, line):
        s = re.split(r'[()]', line)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
            torch.tensor(float(args[2])),
            torch.tensor(float(args[3])),
            torch.tensor(float(args[4])),
            torch.tensor(float(args[5])),
            torch.tensor(float(args[6])),
            torch.tensor(float(args[7]))
        )

            
    # Execute an attach line, creates two attachment points, then figures out how to best satisfy new constraint
    def executeAttach(self, parse):
        ap1 = AttPoint(
            self.cuboids[parse[0]],
            parse[2],
            parse[3],
            parse[4],
        )

        ap2 = AttPoint(
            self.cuboids[parse[1]],
            parse[5],
            parse[6],
            parse[7],
        )

        ap_pt_name = f'{parse[0]}_to_{parse[1]}'
        # Attach points should have unique names
        while ap_pt_name in self.att_points:            
            ap_pt_name += '_n'
        self.att_points[ap_pt_name] = ap2
        
        ap2.cuboid.getAttachments().append((ap2, ap2.getPos(), ap1.cuboid.name))
        self.attach(ap1, ap2.getPos(), ap2.cuboid.name, ap2)

    # Parses a reflect command
    def parseReflect(self, line):
        s = re.split(r'[()]', line)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
        )

    # Parses a translate command
    def parseTranslate(self, line):
        s = re.split(r'[()]', line)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
            int(args[2]),
            float(args[3])
        )

    # Parses a queeze command
    def parseSqueeze(self, line):
        s = re.split(r'[()]', line)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
            args[2],
            args[3],
            float(args[4]),
            float(args[5])
        )

    # Help function for getting direction of reflect commands
    def getRefDir(self, d):
        bbox = self.cuboids['bbox']
        if d == 'X':
            return bbox.rfnorm.clone()
        elif d == 'Y':
            return bbox.tfnorm.clone()
        elif d == 'Z':
            return bbox.ffnorm.clone()
        else:
            assert False, 'bad reflect argument'

    # Help function for getting direction + scale of translate commands
    def getTransDir(self, d):
        bbox = self.cuboids['bbox']
        if d == 'X':
            return bbox.rfnorm.clone(), bbox.dims[0].clone()
        elif d == 'Y':
            return bbox.tfnorm.clone(), bbox.dims[1].clone()
        elif d == 'Z':
            return bbox.ffnorm.clone(), bbox.dims[2].clone()
        else:
            assert False, 'bad reflect argument'
            
    # Given an axis + a center, consructs a tranformation matrix to satisfy reflection
    def getRefMatrixHomo(self, axis, center):

        m = center
        d = axis / axis.norm()

        refmat = torch.stack((
            torch.stack((1 - 2 * d[0] * d[0], -2 * d[0] * d[1], -2 * d[0] * d[2], 2 * d[0] * d[0] * m[0] + 2 * d[0] * d[1] * m[1] + 2 * d[0] * d[2] * m[2])),
            torch.stack((-2 * d[1] * d[0], 1 - 2 * d[1] * d[1], -2 * d[1] * d[2], 2 * d[1] * d[0] * m[0] + 2 * d[1] * d[1] * m[1] + 2 * d[1] * d[2] * m[2])),
            torch.stack((-2 * d[2] * d[0], -2 * d[2] * d[1], 1 - 2 * d[2] * d[2], 2 * d[2] * d[0] * m[0] + 2 * d[2] * d[1] * m[1] + 2 * d[2] * d[2] * m[2]))
        ))

        return refmat

    # Reflect a point p, about center and a direction ndir
    def reflect_point(self, p, center, ndir):
        pad = torch.nn.ConstantPad1d((0, 1), 1.0)
        reflection = self.getRefMatrixHomo(ndir, center)
        posHomo = pad(p)
        return reflection @ posHomo
    
    # Given angle, axis and center constructs a transformation matrix to satisfy rotation
    def getRotMatrixHomo(self, angle, axis, center):
        s = torch.sin(angle)
        c = torch.cos(angle)

        naxis = axis / axis.norm()
        nx = naxis[0]
        ny = naxis[1]
        nz = naxis[2]

        tmpmat = torch.stack((
            torch.stack((c + (1 - c) * nx * nx, (1 - c) * nx * ny - s * nz, (1 - c) * nx * nz + s * ny)),
            torch.stack(((1 - c) * nx * ny + s * nz, c + (1 - c) * ny * ny, (1 - c) * ny * nz - s * nx)),
            torch.stack(((1 - c) * nx * nz - s * ny, (1 - c) * ny * nz + s * nx, c + (1 - c) * nz * nz))
        ))

        rotmat = torch.cat((tmpmat, torch.zeros(3,1)), dim = 1)
        lc = (torch.matmul(torch.eye(3) - tmpmat, center)).unsqueeze(1)
        return torch.cat((rotmat[:,:3], lc), dim = 1)
    
    # Executes a reflect line by making + executing new Cuboid and attach lines
    def executeReflect(self, parse):
        c = self.cuboids[parse[0]]        
        assert c.name != "bbox", 'tried to move the bbox'
        
        rdir = self.getRefDir(parse[1])
        name = c.name + '_ref'
        
        self.executeCuboid([f'{name}', c.dims[0].clone(), c.dims[1].clone(), c.dims[2].clone(), c.aligned])
                        
        self.cuboids[f'{name}'].parent = c.name
        self.cuboids[f'{name}'].parent_axis = parse[1]
        
        atts = c.move_atts
        for att in atts:
            
            if parse[1] == 'X':
                x = 1 - att[0].x.clone()
            else:
                x = att[0].x.clone()

            if parse[1] == 'Y':
                y = 1 - att[0].y.clone()
            else:
                y = att[0].y.clone()

            if parse[1] == 'Z':
                z = 1 - att[0].z.clone()
            else:
                z = att[0].z.clone()
            
            n = att[2]

            cpt = att[0].getPos().clone()
            rpt = self.reflect_point(cpt, self.cuboids['bbox'].pos.clone(), rdir)
            
            rrpt = self.cuboids[n].getRelPos(rpt, True)
            
            self.executeAttach([f'{name}', f'{n}', x, y, z, rrpt[0], rrpt[1], rrpt[2]])
            
    # Executes a translate line by making + executing new Cuboid and attach lines
    def executeTranslate(self, parse):
        
        c = self.cuboids[parse[0]]
        assert c.name != "bbox", 'tried to move the bbox'
        tdir, td = self.getTransDir(parse[1])

        N = parse[2]
        scale = (td * parse[3]) / float(N)

        for i in range(1, N+1):
        

            name = c.name + f'_trans_{i}'
            self.executeCuboid([f'{name}', c.dims[0].clone(), c.dims[1].clone(), c.dims[2].clone(), c.aligned]) 
            self.cuboids[f'{name}'].parent = c.name
            
            atts = c.move_atts
            for att in atts:
                x = att[0].x
                y = att[0].y
                z = att[0].z
                n = att[2]

                cpt = att[0].getPos()
                rpt = cpt + (tdir * scale * i)

                rrpt = self.cuboids[n].getRelPos(rpt, True)

                self.executeAttach([f'{name}', f'{n}', x, y, z, rrpt[0], rrpt[1], rrpt[2]])


    # Helper function for finding opposite face
    def getOppFace(self, face):
        of = {
            'right': 'left',
            'left': 'right',
            'top': 'bot',
            'bot': 'top',
            'front': 'back',
            'back': 'front',
        }
        return of[face]

    # Local coordinate frame to center of face conversion
    def getFacePos(self, face):
        ft = {
            'right': ([1.0, 0.5, 0.5], 0, 0.),
            'left': ([0.0, 0.5, 0.5], 0, 1.),
            'top': ([.5, 1.0, 0.5], 1, 0.),
            'bot': ([.5, 0.0, 0.5], 1, 1.),
            'front': ([.5, 0.5, 1.0], 2, 0.),
            'back': ([.5, 0.5, 0.0], 2, 1.),
        }
        return ft[face]

    # Converts squeeze parameters into parameters needed for the two attachment operators.
    def getSqueezeAtt(self, face, u, v, is_bbox):
        at1, ind, val = self.getFacePos(face)
        # bbox is "flipped"
        if is_bbox:
            rval = 1-val
        else:
            rval = val
        at2 = torch.zeros(3).float()
        q = [u, v] 
        for i in range(3):
            if i == ind:
                at2[i] = rval
            else:
                at2[i] = q.pop(0)

        return torch.tensor(at1).float(), at2

    # Executes a squeeze line by making + executing new Cuboid and attach lines
    def executeSqueeze(self, parse):
        face = parse[3]
        oface = self.getOppFace(face)

        atc1, ato1 = self.getSqueezeAtt(
            face, parse[4], parse[5], parse[1] == 'bbox'
        )

        atc2, ato2 = self.getSqueezeAtt(
            oface, parse[4], parse[5], parse[2] == 'bbox'
        )        
            
        self.executeAttach([parse[0], parse[1], atc1[0], atc1[1], atc1[2], ato1[0], ato1[1], ato1[2]])
        self.executeAttach([parse[0], parse[2], atc2[0], atc2[1], atc2[2], ato2[0], ato2[1], ato2[2]])

    # Clear cuboids + attachment points, but keep the commands that made them in memory
    def resetState(self):
        self.cuboids = self.getBoundBox()
        self.att_points = {}
        
    # Supported commands and their execution functions
    # Commands are first parsed to get their type + parameters. Then, the line is executed by calling to the appropriate execute function 
    def execute(self, line):
        res = None
        if "Cuboid(" in line:
            parse = self.parseCuboid(line)
            self.executeCuboid(parse)
        
        elif "attach(" in line:
            parse = self.parseAttach(line)
            self.executeAttach(parse)    

        elif "reflect(" in line:
            parse = self.parseReflect(line)
            res = self.executeReflect(parse)

        elif "translate(" in line:
            parse = self.parseTranslate(line)
            res = self.executeTranslate(parse)

        elif "squeeze(" in line:
            parse = self.parseSqueeze(line)
            res = self.executeSqueeze(parse)

        # return any new lines generated by macros
        return res
            
    # To re-run a program given a set of commands and parameters. Often used during fitting to unstructurd geometry. 
    def runProgram(self, param_lines):
        self.resetState()

        command_to_func = {
            "Cuboid": self.executeCuboid,
            "attach": self.executeAttach,
            "squeeze": self.executeSqueeze,
            "translate": self.executeTranslate,
            "reflect": self.executeReflect
        }
        
        for command, parse in param_lines:
            func = command_to_func[command]
            func(parse)

# ** Helper Functions FOR ShapeAssembly Class **

def lineToAttrs(line):
    P = Program()
    if "Cuboid(" in line:
        func = "Cuboid"
        parse = list(P.parseCuboid(line))
        param_inds = [1,2,3]                
        
    elif "attach(" in line:
        func = "attach"
        parse = list(P.parseAttach(line))
        param_inds = [2,3,4,5,6,7]
        
    elif "reflect(" in line:
        func = "reflect"
        parse = list(P.parseReflect(line))
        param_inds = []
        
    elif "translate(" in line:
        func = "translate"
        parse = list(P.parseTranslate(line))
        param_inds = [3]

    elif "squeeze(" in line:
        func = "squeeze"
        parse = list(P.parseSqueeze(line))
        param_inds = [4,5]
        
    tensor = torch.nn.Parameter(
        torch.tensor([parse[i] for i in param_inds])
    )
    for i,j in enumerate(param_inds):
        parse[j] = tensor[i] 

    return func, tensor, parse
        
def make_hier_prog(lines):
    all_progs = {}
    root_name = None
        
    cur_name = None
    cur_prog = []
    cur_children = []
        
    for line in lines:
        if len(line) == 0:
            continue
        ls = line.strip().split()

        if ls[0] == 'Assembly':
            cur_name = ls[1]
            if root_name is None:
                root_name = cur_name
                
        elif ls[0] == '}':
            all_progs[cur_name] = (cur_prog, cur_children)
            cur_children = []
            cur_prog = []
            cur_name = None

        else:
            if 'Cuboid' in line:
                if 'Program_' in line:
                    cur_children.append(ls[0])
                else:
                    cur_children.append(None)
                        
            cur_prog.append(line[1:-1])

    hp = {'name': root_name}
    
    q = [hp]

    while(len(q)) > 0:
        node = q.pop(0)
        prog, children = all_progs[node['name']]
        node['prog'] = prog
        node['children'] = []
            
        for child in children:
            c = {}
            if child is not None:
                c = {'name': child}
                q.append(c)
            node['children'].append(c)

    return hp

# Logic to create function text
def make_function(name, args):
    args = [str(arg) for arg in args]
    return '{}({})'.format(name, ", ".join(args))

def assign(var_name, value):
    return '{} = {}'.format(var_name, value)

def get_pos_delta(abox, rbox, pos):

    r = torch.stack((
        abox.rfnorm,
        abox.tfnorm,
        abox.ffnorm
    )).T

    return (r @ pos) + (abox.pos - rbox.pos) 


# Given a cuboid cube, and its local program bounding volume rbox, and the actual placement of its bonding volume abox, find the correct transformation for cube
def apply_delta(abox, rbox, cube):

    r = torch.stack((
        abox.rfnorm,
        abox.tfnorm,
        abox.ffnorm
    )).T

    cube.dims *=  (abox.dims / rbox.dims)    
    cube.pos = (r @ cube.pos) + (abox.pos - rbox.pos) 

    cube.rfnorm = r @ cube.rfnorm
    cube.tfnorm = r @ cube.tfnorm
    cube.ffnorm = r @ cube.ffnorm
    
# Execute a hierarchical shapeassembly program, in a differentiable fashion
def diff_hier_execute(root, param_dict, return_all = False):
           
    q = [(root, None, False)]
       
    scene = []
    
    while len(q) > 0:    
        node, bbox, flip_axis = q.pop(0)

        param_lines = param_dict[node['name']]

        if bbox is None:
            bbox = Cuboid("bbox", aligned = True, vis=False)
            bbox.dims = torch.stack(param_lines[0][1][1:4])
            
        TP = Program()
        TP.runProgram(param_lines)

        if flip_axis:
            TP.flip(flip_axis)
        
        rbox = TP.cuboids.pop('bbox')
    
        add = []

        for i, c_key in enumerate(TP.cuboids.keys()):
            flip = False
            cub = TP.cuboids[c_key]
            child = None
            
            if i+1 < len(node["children"]):
                child = node["children"][i+1]
                
            elif cub.parent is not None:
                pi = list(TP.cuboids.keys()).index(cub.parent)
                child = deepcopy(node["children"][pi+1])
                if cub.parent_axis is not None and 'prog' in child:
                    flip_axis = cub.parent_axis

            # cub is found through local execution, this brings it into global space
            apply_delta(bbox, rbox, cub)
            
            # if intermediate cuboid, add back into queue
            if child is not None and len(child) > 0:                    
                q.append((child, cub, flip_axis))
            # if leave cuboid, save these cuboid to the add list
            else:
                add.append(cub)
                
        scene += add
            
    verts = torch.tensor([],dtype=torch.float)
    faces = torch.tensor([],dtype=torch.long)
    
    for cube in scene:
        v, f = cube.getTris()
        if v is not None and f is not None:
            faces =  torch.cat((faces, (f + verts.shape[0])))
            verts = torch.cat((verts, v))


    if return_all:        
        return verts, faces, scene
            
    return verts, faces

    
# Execute a hierarchical shapeassembly program
def hier_execute(root, return_all=False):
           
    bbox = Cuboid('bbox')
        
    bbox.dims = torch.tensor(
        [float(a) for a in re.split(r'[()]', root['prog'][0])[1].split(',')[:3]]
    )
    
    q = [(root, bbox, None)]
       
    scene = []
    count = 0
    
    while len(q) > 0:    
        node, bbox, flip_axis = q.pop(0)

        lines = node["prog"]
        TP = Program()

        for line in lines:
            TP.execute(line)        

        if flip_axis:
            TP.flip(flip_axis)
            
        rbox = TP.cuboids.pop('bbox')
    
        add = []

        i_map = {}
        i = 0
        
        for c_key in TP.cuboids.keys():
            cub = TP.cuboids[c_key]
            child = None
            flip_axis = False

            if '_ref' not in c_key and '_trans' not in c_key:
                child = node["children"][i+1]
                i_map[c_key] = i
                i += 1
            else:
                pi = i_map[cub.parent]
                child = deepcopy(node["children"][pi+1])
                if cub.parent_axis is not None and 'prog' in child:
                    flip_axis = cub.parent_axis                
                    
            # cub is found through local execution, this brings it into global space
            apply_delta(bbox, rbox, cub)
            # if intermediate cuboid, add back into queue
            if child is not None and len(child) > 0:                    
                q.append((child, cub, flip_axis))
            # if leave cuboid, save these cuboid to the add list
            else:
                add.append(cub)
                
        scene += add
        
    verts = torch.tensor([],dtype=torch.float)
    faces = torch.tensor([],dtype=torch.long)
    
    for cube in scene:
        v, f = cube.getTris()
        if v is not None and f is not None:
            faces =  torch.cat((faces, (f + verts.shape[0])))
            verts = torch.cat((verts, v))

    if not return_all:    
        return verts, faces
    else:
        return verts, faces, scene

# Execute a hierarchical shapeassembly program
def hier_execute_to_npz(root):
           
    bbox = Cuboid('bbox')
        
    bbox.dims = torch.tensor(
        [float(a) for a in re.split(r'[()]', root['prog'][0])[1].split(',')[:3]]
    )
    
    q = [(root, bbox, None)]
       
    scene_cubes = []
    scene_aps = []
    count = 0
    
    while len(q) > 0:    
        node, bbox, flip_axis = q.pop(0)

        lines = node["prog"]
        TP = Program()

        for line in lines:
            TP.execute(line)        

        if flip_axis:
            TP.flip(flip_axis)
            
        rbox = TP.cuboids.pop('bbox')
    
        add_cubes = []
        add_aps = [
            (
                ap,
                get_pos_delta(bbox, rbox, TP.att_points[ap].getPos())
            )
            for ap in TP.att_points
        ]
        
        i_map = {}
        i = 0
        remove_inds = set()
        for c_key in TP.cuboids.keys():
            cub = TP.cuboids[c_key]
            child = None
            flip_axis = False

            if '_ref' not in c_key and '_trans' not in c_key:
                child = node["children"][i+1]
                i_map[c_key] = i
                i += 1
            else:
                pi = i_map[cub.parent]
                child = deepcopy(node["children"][pi+1])
                if cub.parent_axis is not None and 'prog' in child:
                    flip_axis = cub.parent_axis                
                    
            # cub is found through local execution, this brings it into global space
            apply_delta(bbox, rbox, cub)
            # if intermediate cuboid, add back into queue
            if child is not None and len(child) > 0:
                for j, (ap_name, _) in enumerate(add_aps):
                    remove_inds.add(j)
                q.append((child, cub, flip_axis))
            # if leave cuboid, save these cuboid to the add list
            else:
                add_cubes.append(cub)
                
        scene_cubes += add_cubes
        scene_aps += [
            pos for i, (_, pos) in enumerate(add_aps) if i not in remove_inds
        ]
        
    scene_cubes = [
        torch.cat((
            c.pos, c.dims, c.rfnorm, c.tfnorm
        )).numpy() for c in scene_cubes
    ]

    scene_cubes = np.array(scene_cubes)

    scene_aps = [ap.numpy() for ap in scene_aps]
    scene_aps = np.array(scene_aps)
    
    return scene_cubes, scene_aps
            
class ShapeAssembly():
    """ Contains entrypoint logic for parsing and executing entire programs"""
    
    # Execute a program without any hierarchy
    def run_local(self, lines, out_file):
        P = Program()
        for line in lines:
            P.execute(line)
        P.render(out_file)

    # Execute a program differentiable w.r.t. to the parameters in the param_dict
    def diff_run(self, hier, param_dict):
        return diff_hier_execute(hier, param_dict)
        
    # Execute a program 
    def run(self, lines, out_file):
        hier_prog = make_hier_prog(lines)
        verts, faces = hier_execute(hier_prog)
        writeObj(verts, faces, out_file)

    # load lines from a program file
    def load_lines(self, prog_file):
        lines = []
        with open(prog_file) as f:
            for line in f:
                lines.append(line)
        return lines                

    # Convert a hierarchy + dictionary of parameters into a full ShapeAssembly program
    def fill_hier(self, hier, param_dict):
        q = [hier]
        while(len(q) > 0):
            node = q.pop(0)
            param_lines = param_dict.pop(node['name'])
            lines = []
            for func, tparams in param_lines:
                params = [
                    round(p.item(), 2) if isinstance(p, torch.Tensor) else p \
                    for p in tparams
                ]
                if func == "Cuboid":
                    lines.append(
                        assign(
                            params[0],
                            make_function(func, params[1:])
                        )
                    )
                else:
                    lines.append(
                        make_function(func, params)
                    )
            node['prog'] = lines

            for c in node['children']:
                if len(c) > 0:
                    q.append(c)
        
    # Return a program hierarchy, a dictionary of
    # nodes to lines (as tensors), and a list
    # of all tensors in lines -> used during differentiable execution
    def make_hier_param_dict(self, lines):
        h = make_hier_prog(lines)
        q = [h]

        param_list = []
        param_dict = {}
        
        while len(q) > 0:
            node = q.pop(0)
            prog = node.pop('prog')
            lines = []

            for line in prog:                
                func, tensor, parse = lineToAttrs(line)
                lines.append((func, parse))
                if tensor.shape[0] > 0:
                    param_list.append(tensor)

            param_dict[node['name']] = lines

            for c in node['children']:
                if len(c) > 0:
                    q.append(c)

        return h, param_dict, param_list
        
                
        
if __name__ == '__main__':
    mode, prog_file, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
    sa = ShapeAssembly()
    lines = sa.load_lines(prog_file)
    if mode == 'run':
        sa.run(lines, out_file)
    elif mode == 'run_local':
        sa.run_local(lines, out_file)
    else:
        print(f'Mode {mode} not recognized')
