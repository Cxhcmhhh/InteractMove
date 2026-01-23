import sys
import os
sys.path.append(os.path.abspath('./'))
import random
import numpy as np
import trimesh
import pickle
import json
import glob
from collections import defaultdict
from plyfile import PlyData, PlyElement
from pyquaternion import Quaternion as Q
from utils.visualization import frame2video, grab_render_motion_in_scene
from utils.smplx_util import SMPLX_Util
from sklearn.neighbors import KDTree
from utils.geo_utils import is_point_in_cuboid, make_M_from_tqs, create_unit_bbox, create_vector, rotate_2D_points_along_z_axis
import utils.configuration as config
from dataset.sr3d import SR3D_Util

########################################
## align class
########################################
class ActionAlign():
    def __init__(
        self, 
        annotations: dict,
        instance_to_semantic: dict,
        label_mapping: dict,
        scene_path: str, 
        static_scene: trimesh.PointCloud, 
        static_scene_label: np.ndarray,
        static_scene_trans: np.ndarray,
        body_vertices: np.ndarray, 
        joints_traj: np.ndarray,
    ):
        """ Action Align father class
        """
        self.action = None

        self.annotations = annotations
        self.instance_to_semantic = instance_to_semantic
        self.label_mapping = label_mapping

        self.scene_id = scene_path.split('/')[-2]
        self.static_scene = static_scene
        self.static_scene_label = static_scene_label
        self.static_scene_trans = static_scene_trans
        self.body_vertices = body_vertices
        self.joints_traj = joints_traj

        self.scene_occupancy = self.get_scene_occupancy(static_scene_label)
        self.floor_occupancy = self.get_scene_floor_occupancy(static_scene_label)
        self.wall_occupancy = self.get_scene_wall_occupancy(static_scene_label)

        lf_traj, rf_traj = SMPLX_Util.get_foots_trajectory(body_vertices)
        self.left_foot_traj = lf_traj
        self.right_foot_traj = rf_traj
    
    def get_butt_verts(self, f: int=0):
        """ Get butt vertices of f-th frame

        Args:
            f: the index of frame

        Return:
            The butt vertices coordinates in 3D
        """
        return SMPLX_Util.get_butt_verts(self.body_vertices[f])
    
    def get_knee_verts(self, f: int=0):
        """ Get knee vertices of f-th frame

        Args:
            f: the index of frame

        Return:
            The knee vertices coordinates in 3D
        """
        return SMPLX_Util.get_knee_verts(self.body_vertices[f])
    
    def get_right_hand_position(self, f: int=0):
        """ Get right hand position of f-th frame

        Args:
            f: the index of frame

        Return:
            The right hand position coordinates in 3D
        """
        return SMPLX_Util.get_right_hand(self.body_vertices[f]).mean(axis=0)
    
    def get_body_orient(self, f: int=0, xy: bool=False):
        """ Get body orientation of f-th frame

        Args:
            f: the index of frame
        
        Return:
            The body orient
        """
        orient3D = SMPLX_Util.get_body_orient(self.joints_traj[f])

        if xy == True:
            return orient3D[0:2] / np.linalg.norm(orient3D[0:2])
        return orient3D
    
    def get_scene_occupancy(self, scene_labels: np.ndarray):
        """ Get occupied scene indices, objects occupancy (without floor, ceiling, unlabeled)

        Args:
            scene_labels: scene semantic lable array
        
        Return:
            A bool array, True indicate the occupied space
        """
        ## containing floor, floor mat, ceiling, and unlabled vertices
        free_space_indices = (scene_labels == self.label_mapping['floor']) | (scene_labels == self.label_mapping['floor mat']) | (scene_labels == self.label_mapping['ceiling']) | (scene_labels == 0)
        return np.logical_not(free_space_indices)
    
    def get_scene_floor_occupancy(self, scene_labels: np.ndarray):
        """ Get scene floor indices

        Args:
            scene_labels: scene semantic lable array
        
        Return:
            A bool array, True indicate the floor
        """
        floor_space_indices = (scene_labels == self.label_mapping['floor']) | (scene_labels == self.label_mapping['floor mat'])
        return floor_space_indices
    
    def get_scene_wall_occupancy(self, scene_labels: np.ndarray):
        """ Get scene wall indices

        Args:
            scene_labels: scene semantic lable array
        
        Return:
            A bool array, True indicate the wall
        """
        wall_space_indices = (scene_labels == self.label_mapping['wall']) | (scene_labels == self.label_mapping['window'])
        return wall_space_indices
    
    def get_surrounding_height(self, occupied_kdtree: KDTree, verts: np.ndarray, qurey_point_xy: np.ndarray):
        """ Get the height of surrounding floor/object for translating body, since the scannet scene floor may be uneven

        Args:
            occupied_kdtree: the kdtree of 2D floor points
            verts: floor vertices in 3D
            query_point_xy: a query point
        
        Return:
            The height of surrounding floor
        """
        if occupied_kdtree is None:
            return 0
        
        _, indic = occupied_kdtree.query(np.array([[*qurey_point_xy]]), k=20)
        indic = indic[0]
        return verts[indic][:, -1].mean()
    
    def get_valid_interact_object_list(self, related_object_group: list):
        """ Get the list of valid objects

        Args:
            related_object_group: the list of interact obejcts
        
        Return:
            Valid interactive object list and the object occurrence count
        """
        aggregation_file = os.path.join(config.scannet_folder, self.scene_id, self.scene_id + '.aggregation.json')
        segment_file = os.path.join(config.scannet_folder, self.scene_id, self.scene_id + '_vh_clean_2.0.010000.segs.json')

        with open(aggregation_file, 'r') as fp:
            scan_aggregation = json.load(fp)

        with open(segment_file, 'r') as fp:
            segment_info = json.load(fp)
            segment_indices = segment_info['segIndices']
        
        segment_indices_dict = defaultdict(list)
        for i, s in enumerate(segment_indices):
            segment_indices_dict[s].append(i)  # Add to each segment, its point indices
        
        ## iterate over all object
        all_objects = []
        occurrences = defaultdict(int)
        for object_info in scan_aggregation['segGroups']:
            object_instance_label = object_info['label']
            occurrences[object_instance_label] += 1

            semantic_label = self.label_mapping[ self.instance_to_semantic[object_instance_label] ]
            if semantic_label not in related_object_group: # interact with some selected object categories
                continue

            object_id = object_info['objectId']
            segments = object_info['segments']
            pc_loc = []
            for s in segments:
                pc_loc.extend(segment_indices_dict[s])
            object_pc = pc_loc

            all_objects.append((object_id, object_instance_label, semantic_label, object_pc))
        
        return all_objects, occurrences

    
    @staticmethod
    def calc_Mbbox(model: dict):
        """ Get transformation matrix of bounding box of scan2cad format

        Args:
            model: an instance annotation in scan2cad
        
        Return:
            The transformation matrix of the annotated instance
        """
        trs_obj = model["trs"]
        bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
        center_obj = np.asarray(model["center"], dtype=np.float64)
        trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
        q_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
        scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

        tcenter1 = np.eye(4)
        tcenter1[0:3, 3] = center_obj
        trans1 = np.eye(4)
        trans1[0:3, 3] = trans_obj
        rot1 = np.eye(4)
        rot1[0:3, 0:3] = Q(q_obj).rotation_matrix
        scale1 = np.eye(4)
        scale1[0:3, 0:3] = np.diag(scale_obj)
        bbox1 = np.eye(4)
        bbox1[0:3, 0:3] = np.diag(bbox_obj)
        M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
        return M
    
    @staticmethod
    def get_color_array(length, color: np.ndarray=np.array([64, 64, 64, 255], dtype=np.uint8)):
        color = np.ones((length, 4), dtype=np.uint8) * color
        return color
    
    @staticmethod
    def random_indices(length: int, count: int):
        """ Get random indices of range with length

        Args:
            length: length of the range, i.e. 0, ... ,length - 1
            count: sample count
        
        Return:
            indices list
        """
        indic = list(range(length))
        random.shuffle(indic)
        count = min(len(indic), count)
        return indic[0:count]
    
    def sample_proposal(self):
        pass


########################################
## Open align class
########################################
class RaiseAlign(ActionAlign):
    def __init__(self, annotations: dict, instance_to_semantic: dict, label_mapping: dict, scene_path: str, static_scene: trimesh.PointCloud, static_scene_label: np.ndarray, static_scene_trans: np.ndarray, body_vertices: np.ndarray, joints_traj: np.ndarray):
        super(RaiseAlign, self).__init__(annotations, instance_to_semantic, label_mapping, scene_path, static_scene, static_scene_label, static_scene_trans, body_vertices, joints_traj)

        self.action = 'raise'
        self.interact_object_group = [7, 14] # table, desk

        self.TOWARD_ORIENTATION_CONSTRAINT = np.pi / 6

    def _detect_body_in_scene(self, scene_kdtree: KDTree, all_points: np.ndarray, delta: float=0.2):
        """ Detect the points for motion that are in the scene.

        Args:
            scene_kdtree: scene vertices kdtree (2D)
            all_points: all detected points
            delta: default is 0.1
        
        Return:
            A bool value
        """
        dist_obj, _ = scene_kdtree.query(all_points[:, 0:2], k=1)

        out_obj = dist_obj.reshape(-1) > delta
        if out_obj.sum() > 0: # some points are out of the obejct
            return False

        return True
    
    def _detect_collision_with_wall(self, wall_kdtree: KDTree, all_points: np.ndarray, delta: float=0.1):
        """ Detect the points for motion that should have no collision with wall

        Args:
            wall_kdtree: wall vertices kdtree (2D)
            all_points: all detected points
            delta: default is 0.1
        
        Return:
            A bool value
        """
        if wall_kdtree is None:
            return True
        
        dist_obj, _ = wall_kdtree.query(all_points[:, 0:2], k=1)

        in_obj = dist_obj.reshape(-1) < delta
        if in_obj.sum() > 0: # some points are too close to the wall
            return False

        return True
    
    def _detect_collision_with_occupied_scene(self, occupied_scene_kdtree: KDTree, all_points: np.ndarray, delta: float=0.05):
        """ Detect the points for motion that should have no collision with occupied scene

        Args:
            wall_kdtree: occupied scene vertices kdtree (2D)
            all_points: all detected points
            delta: default is 0.05
        
        Return:
            A bool value
        """
        dist_obj, _ = occupied_scene_kdtree.query(all_points[:, 0:2], k=1)

        in_obj = dist_obj.reshape(-1) < delta
        if in_obj.sum() > 0:
            return False

        return True

    def _sample_proposal_for_raise(self, object_verts: np.ndarray, non_object_verts: np.ndarray, bin_n: int=72, delta: float=0.2):
        """ Sample valid position and orientation for raise

        Args:
            object_verts: occupied object vertices
            non_object_verts: non object vertices
            bin_n: orientation bin, default is 72
        
        Returns:
            Valid position of pelvis, Valid rotation of pelvis, 
        """
        object_xy = object_verts[:, 0:2]
        min_x, min_y = object_xy.min(axis=0)
        max_x, max_y = object_xy.max(axis=0)
        object_center_xy = object_xy.mean(axis=0)

        object_kdtree = KDTree(object_verts[:, 0:2], leaf_size=int(len(object_verts) * 0.8))
        non_object_kdtree = KDTree(non_object_verts[:, 0:2], leaf_size=int(len(non_object_verts) * 0.8))

        scene_KDtree = KDTree(self.static_scene.vertices[:, 0:2], leaf_size=int(len(self.static_scene.vertices) * 0.8)) # whole scene
        scene_occupied_verts = self.static_scene[self.scene_occupancy]
        scene_occupied_KDtree = KDTree(scene_occupied_verts[:, 0:2], leaf_size=int(len(scene_occupied_verts) * 0.8))
        floor_occupied_verts = self.static_scene[self.floor_occupancy]
        floor_occupied_KDtree = None if len(floor_occupied_verts) == 0 else KDTree(floor_occupied_verts[:, 0:2], leaf_size=int(len(floor_occupied_verts) * 0.8))
        wall_occupied_verts = self.static_scene[self.wall_occupancy]
        wall_occupied_KDtree = None if len(wall_occupied_verts) == 0 else KDTree(wall_occupied_verts[:, 0:2], leaf_size=int(len(wall_occupied_verts) * 0.8))
        

        ## sample grid
        proposed_points = []
        for x in np.arange(min_x - delta, max_x + delta, delta * 0.5):
            for y in np.arange(min_y - delta, max_y + delta, delta * 0.5):
                point = np.array([[x, y]])

                dist_obj, _ = object_kdtree.query(point, k=1)
                dist_non_obj, _ = non_object_kdtree.query(point, k=1)
                
                if floor_occupied_KDtree is not None:
                    dist_floor, _ = floor_occupied_KDtree.query(point, k=1)
                    if dist_floor[0][0] > delta:
                        continue 

                if dist_obj[0][0] < delta and dist_obj[0][0] > 0.5 * delta and dist_non_obj[0][0] > delta:
                    proposed_points.append(point.reshape(-1))


        ## filter proposed position
        body_orient_xy_last = self.get_body_orient(-1, xy=True)
        pelvis_xy_rotate = self.joints_traj[-1, 0, 0:2] # last frame pelvis as anchor

        valid_trans = []
        valid_orient = []
        debug_body_orient = []
        debug_obj_orient = []
        for i, xy, in enumerate(proposed_points):
            body_trans_z = self.get_surrounding_height(floor_occupied_KDtree, floor_occupied_verts, xy)
            trans = np.array([*xy, body_trans_z], dtype=np.float32)

            for angle in np.arange(0, np.pi * 2, np.pi * 2 / bin_n):

                ## last frame of body must toward the obejct
                toward_direction = rotate_2D_points_along_z_axis(body_orient_xy_last, angle)
                target_direction = object_center_xy - xy
                target_direction = target_direction / np.linalg.norm(target_direction)
                if np.dot(toward_direction, target_direction) < np.cos(self.TOWARD_ORIENTATION_CONSTRAINT):
                    continue

                ## every frame of body must be in the scene
                all_points = self.joints_traj[:, 0, :] - np.array([*pelvis_xy_rotate, 0]) # use all pelvis points
                all_points = all_points.reshape(-1, 3)
                all_points[:, 0:2] = rotate_2D_points_along_z_axis(all_points[:, 0:2], angle)
                all_points += trans
                if not self._detect_body_in_scene(scene_KDtree, all_points):
                    continue

                ## every frame of body should have no collision with wall (specific detection)
                all_points = SMPLX_Util.get_sparse_skeleton(self.joints_traj) - np.array([*pelvis_xy_rotate, 0]) # use skeleton points
                all_points = all_points.reshape(-1, 3)
                all_points[:, 0:2] = rotate_2D_points_along_z_axis(all_points[:, 0:2], angle)
                all_points += trans
                if not self._detect_collision_with_wall(wall_occupied_KDtree, all_points):
                    continue

                ## every frame of body should have no collision with scene, consider foot points
                all_points = np.vstack((
                    self.left_foot_traj.reshape(-1, 3),
                    self.right_foot_traj.reshape(-1, 3),
                ))
                all_points -= np.array([*pelvis_xy_rotate, 0])
                all_points[:, 0:2] = rotate_2D_points_along_z_axis(all_points[:, 0:2], angle)
                all_points += trans
                if not self._detect_collision_with_occupied_scene(scene_occupied_KDtree, all_points):
                    continue


                valid_trans.append(trans)
                valid_orient.append(angle)
                debug_body_orient.append(rotate_2D_points_along_z_axis(body_orient_xy_last, angle))
                debug_obj_orient.append(None)
        
        return np.array(valid_trans), np.array(valid_orient), np.array(debug_body_orient), np.array(debug_obj_orient)

    def sample_proposal(self, sr3d, correct_orient_option: str=None, max_s_per_object: int=5, use_lang: bool=False):
        """ Sample valid position and orientation for raise action
        """
        proposed_trans = []
        proposed_orient = []
        proposed_utterance = []
        proposed_object_id = []
        proposed_object_label = []
        proposed_object_semantic_label = []
        proposed_object_indice = []
        debug_body_o = []
        debug_obj_o = []

        scene_minx, scene_miny = self.static_scene.vertices[:, 0:2].min(axis=0)
        scene_maxx, scene_maxy = self.static_scene.vertices[:, 0:2].max(axis=0)
        scene_max_size = max(scene_maxx - scene_minx, scene_maxy - scene_miny)
        traj_len = np.sqrt(((self.joints_traj[-1, 0, 0:2] - self.joints_traj[0, 0, 0:2]) ** 2).sum())
        if traj_len > scene_max_size:
            ## scene is too small
            return proposed_trans, proposed_orient, proposed_utterance, proposed_object_id, proposed_object_label, proposed_object_semantic_label, proposed_object_indice, debug_body_o, debug_obj_o


        object_list, occurrences = self.get_valid_interact_object_list(self.interact_object_group)

        for obj in object_list:
            if (proposed_trans):
                break
            object_id, object_label, object_semantic, object_pc_indices = obj

            ## if use language annotation, the occurrence of the object must be 1 or sr3d contians utterance
            ## or continue
            if use_lang and occurrences[object_label] != 1 and not sr3d.contain_utterance(self.scene_id, object_id):
                continue


            object_occupancy = np.zeros(len(self.static_scene.vertices), dtype=bool)
            object_occupancy[object_pc_indices] = True
            object_verts = self.static_scene.vertices[object_occupancy]
            non_object_verts = self.static_scene.vertices[self.scene_occupancy & (~object_occupancy)] # get non-object verts (without floor)


            [valid_trans, valid_orient, debug_body_orient, debug_obj_orient] = self._sample_proposal_for_raise(
                object_verts,
                non_object_verts,
                bin_n=36,
            )


            ## select max_s_per_object valid position for each object
            indic = self.random_indices(len(valid_trans), max_s_per_object)

            ## generate utterance for interact obejct
            if use_lang:
                valid_utterance = sr3d.generate_utterance(self.action, self.scene_id, object_id, object_label, len(indic))
            else:
                valid_utterance = [""] * len(indic)
            
            for i, ind in enumerate(indic):
                proposed_trans.append(valid_trans[ind])
                proposed_orient.append(valid_orient[ind])
                proposed_utterance.append(valid_utterance[i])
                proposed_object_id.append(object_id)
                proposed_object_label.append(object_label)
                proposed_object_semantic_label.append(object_semantic)
                proposed_object_indice.append(object_pc_indices)
                debug_body_o.append(debug_body_orient[ind])
                debug_obj_o.append(debug_obj_orient[ind])
    
        return proposed_trans, proposed_orient, proposed_utterance, proposed_object_id, proposed_object_label, proposed_object_semantic_label, proposed_object_indice, debug_body_o, debug_obj_o



########################################
## Alignment main class
########################################
class Alignment():
    def __init__(self, scan2cad_anno: str, sr3d_csv: str):
        with open(scan2cad_anno, 'r') as fp:
            full_annotations = json.load(fp)
        self.annotations = {}
        for a in full_annotations:
            id_scan = a['id_scan']
            self.annotations[id_scan] = a
        print('load scan2cad done...')
    
        self.sr3d = SR3D_Util(sr3d_csv)    
        print('load sr3d done...')
        print()
        
        with open('./dataset/data/scannet_instance_class_to_semantic_class.json', 'r') as fp:
            self.instance_to_semantic = json.load(fp)
        
        with open('./dataset/data/label_mapping.json', 'r') as fp:
            self.label_mapping = json.load(fp)
    
    def read_ply(self, ply_p: str):
        """ Read scanned scene model from .ply data. For different dataset, need to overrider this function.

        Args:
            ply_p: .ply model path

        Return:
            A trimesh.Trimesh of scene and the semantic label array of scene vertices.
        """
        with open(ply_p, 'rb') as f:
            plydata = PlyData.read(f)
        
            num_verts = plydata['vertex'].count
            # vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
            # colors = np.zeros(shape=[num_verts, 4], dtype=np.uint8)
            labels = np.zeros(shape=num_verts, dtype=np.ushort)

            # vertices[:, 0] = plydata['vertex'].data['x']
            # vertices[:, 1] = plydata['vertex'].data['y']
            # vertices[:, 2] = plydata['vertex'].data['z']

            # colors[:, 0] = plydata['vertex'].data['red']
            # colors[:, 1] = plydata['vertex'].data['green']
            # colors[:, 2] = plydata['vertex'].data['blue']
            # colors[:, 3] = plydata['vertex'].data['alpha']

            labels[:] = plydata['vertex'].data['label']
        
        scene = trimesh.load(ply_p, process=False)
        verts = scene.vertices
        color = scene.visual.vertex_colors

        return trimesh.PointCloud(vertices=verts, colors=color), labels
    
    def get_scene_floor_occupancy(self, scene_labels: np.ndarray):
        """ Get scene floor indices

        Args:
            scene_labels: scene semantic lable array
        
        Return:
            A bool array, True indicate the floor
        """
        floor_space_indices = (scene_labels == self.label_mapping['floor']) | (scene_labels == self.label_mapping['floor mat'])
        return floor_space_indices

    def transform_smplx_from_origin_to_sampled_position(
        self,
        sampled_trans: np.ndarray,
        sampled_rotat: np.ndarray,
        origin_trans: np.ndarray,
        origin_orient: np.ndarray,
        origin_pelvis: np.ndarray
    ):
        """ Convert original smplx parameters to transformed smplx parameters

        Args:
            sampled_trans: sampled valid position
            sampled_rotat: sampled valid rotation
            origin_trans: original trans param array
            origin_orient: original orient param array
            origin_pelvis: original pelvis trajectory
        
        Return:
            Transformed trans, Transformed orient, Transformed pelvis
        """
        position = sampled_trans
        rotat = sampled_rotat

        T1 = np.eye(4, dtype=np.float32)
        T1[0:2, -1] = -origin_pelvis[self.ANCHOR_FROME, 0:2]
        T2 = Q(axis=[0, 0, 1], angle=rotat).transformation_matrix.astype(np.float32)
        T3 = np.eye(4, dtype=np.float32)
        T3[0:3, -1] = position
        T = T3 @ T2 @ T1

        trans_t = []
        orient_t = []
        pelvis_t = []
        for i in range(len(origin_trans)):
            t_, o_ = SMPLX_Util.convert_smplx_verts_transfomation_matrix_to_body(T, origin_trans[i], origin_orient[i], origin_pelvis[i])
            trans_t.append(t_)
            orient_t.append(o_)
            pelvis_t.append(position)
        trans_t = np.array(trans_t)
        orient_t = np.array(orient_t)
        pelvis_t = np.array(pelvis_t)
        return trans_t, orient_t, pelvis_t
    
    def _get_anchor_frame_index(self, action: str):
        if action == 'sit':
            return -1
        elif action == 'stand up':
            return 0
        elif action == 'walk':
            return -1
        elif action == 'lie':
            return -1
        elif action == 'jump':
            return -1
        elif action == 'turn':
            return -1
        elif action == 'place something':
            return -1
        elif action == 'raise':
            return -1
        elif action == 'knock':
            return 0
        elif action == 'dance':
            return 0
        else:
            raise Exception('Unexcepted action type.')

    def align_motion_portal(self, 
        scene_path: str, 
        motion_path: str, 
        action: str,
        anno_path: str,
        max_s: int=5,
        save_res: bool=False,
        visualize: bool=False,
        rendering: bool=False,
        use_lang: bool=False,
    ):
        """ The portal of motion alignment

        Args:
            scene_path:
            motion_path:
            action:
            anno_path:
            max_s: max number of smapled position and orientation pairs for per object
            save_res: save proposal as .pkl file
            visualize: visualize for debug
            rendering: rendering video

        Return:
            The number of valid proposal
        """
        self.ANCHOR_FROME = self._get_anchor_frame_index(action)

        ## load scene
        print("load scene")
        static_scene, static_scene_label = self.read_ply(scene_path)

        floor_occupancy = self.get_scene_floor_occupancy(static_scene_label)
        floor_verts = static_scene.vertices[floor_occupancy]
        if len(floor_verts) == 0: # some scene doesn't have floor
            translate_mat = np.zeros(3, dtype=np.float32)
        else:
            translate_mat = -floor_verts.mean(axis=0)
        static_scene.apply_translation(translate_mat) # move scene to origin center, floor approximately equals to xy plane

        ## load motion
        print("load motion")
        with open(motion_path, 'rb') as fp:
            _, trans, orient, body_pose, left_hand_pose, right_hand_pose, n_comps, _ = pickle.load(fp)
        body_vertices, body_faces, joints_traj = SMPLX_Util.get_grab_body_vertices_sequence(config.smplx_folder, (trans, orient, body_pose, left_hand_pose, right_hand_pose, n_comps))

        ## sample valid position and rotation according to sit action
        print("sampling")
        if action == 'raise':
            action_align = RaiseAlign(self.annotations, self.instance_to_semantic, self.label_mapping, scene_path, static_scene, static_scene_label, translate_mat, body_vertices, joints_traj)
        else:
            raise Exception('Unsupport action: {}'.format(action))
        
        [proposed_trans, proposed_orient, proposed_utterance, proposed_object_id, proposed_object_label, proposed_object_semantic_label,
        proposed_object_indice, debug_body_o, debug_obj_o] = action_align.sample_proposal(
            self.sr3d,
            correct_orient_option='auto',
            max_s_per_object=max_s,
            use_lang=use_lang,
        )

        ## saved data
        proposal = []
        for ind in range(len(proposed_trans)):
            proposal.append({
                'action': action,
                'motion': motion_path.split('/')[-2],
                'scene': scene_path.split('/')[-2],
                'scene_translation': translate_mat,
                'translation': proposed_trans[ind],
                'rotation': proposed_orient[ind],
                'utterance': proposed_utterance[ind],
                'object_id': proposed_object_id[ind],
                'object_label': proposed_object_label[ind],
                'object_semantic_label': proposed_object_semantic_label[ind],
                'object_indice': proposed_object_indice[ind]
            })
            # print(proposed_utterance[ind])
            
        if save_res and len(proposal) != 0:
            os.makedirs(os.path.dirname(anno_path), exist_ok=True)
            with open(anno_path, 'wb') as fp:
                pickle.dump(proposal, fp)
        '''
        ## visualization for debug
        if visualize:
            for i, p in enumerate(proposal):
                self.visualize_debug(
                    static_scene,
                    p['translation'],
                    p['rotation'],
                    body_vertices,
                    joints_traj[:, 0, :],
                    body_faces,
                    debug_body_o[i],
                    debug_obj_o[i],
                )
        '''
        ## rendering video
        if rendering:
            scene_mesh = trimesh.load(scene_path, process=False)
            print('rendering begins')
            scene_mesh.apply_translation(translate_mat)
            for i, p in enumerate(proposal):
                posit = p['translation']
                rotat = p['rotation']
                ascene_mesh = scene_mesh
                obj_occupancy = np.zeros(len(scene_mesh.vertices), dtype=bool)
                obj_occupancy[p['object_indice']] = True
                obj_occupancy = ~obj_occupancy
                facemask = obj_occupancy[scene_mesh.faces].all(axis=1)
                ascene_mesh.update_faces(facemask)
                # ascene_mesh.update_vertices(obj_occupancy)
                trans_, orient_, _ = self.transform_smplx_from_origin_to_sampled_position(posit, rotat, trans, orient, joints_traj[:, 0, :])
                print("prepare rendering")
                self.rendering(
                    anno_path,
                    (trans_, orient_, body_pose, left_hand_pose, right_hand_pose, n_comps),
                    ascene_mesh,
                    i
                )

        return len(proposal)

    def visualize_debug(
        self, 
        static_scene: trimesh.PointCloud,
        position: np.ndarray,
        rotation: np.ndarray,
        body_vertices: np.ndarray,
        pelvis_traj: np.ndarray,
        body_faces: np.ndarray,
        debug_body_o: np.ndarray=None,
        debug_obj_o: np.ndarray=None,
    ):
        """ Visualization for debug
        """
        scene_pc = trimesh.PointCloud(vertices=static_scene.vertices, colors=static_scene.visual.vertex_colors)

        sampled_points = np.zeros((len(position), 3)) - 0.05
        sampled_points[:, 0:2] = position[0:2]
        color = np.ones((len(sampled_points), 4), dtype=np.uint8) * 255
        color[:, 0:3] = np.array([255, 0, 0])

        s = trimesh.Scene()
        s.add_geometry(scene_pc)
        s.add_geometry(trimesh.PointCloud(vertices=sampled_points, colors=color))

        for i in range(0, len(body_vertices), 10):
            body_verts = body_vertices[i].copy()
            body_verts[:, 0:2] -= pelvis_traj[self.ANCHOR_FROME, 0:2]
            body_verts[:, 0:2] = rotate_2D_points_along_z_axis(body_verts[:, 0:2], rotation)
            body_verts += position
            s.add_geometry(trimesh.Trimesh(body_verts, body_faces, process=False))
        body_verts = body_vertices[-1].copy()
        body_verts[:, 0:2] -= pelvis_traj[self.ANCHOR_FROME, 0:2]
        body_verts[:, 0:2] = rotate_2D_points_along_z_axis(body_verts[:, 0:2], rotation)
        body_verts += position
        s.add_geometry(trimesh.Trimesh(body_verts, body_faces, process=False))

        if debug_body_o is not None:
            body_o_v = create_vector(orient_vec=np.array([*debug_body_o, 0]))
            body_o_v.apply_translation(np.array([*position[0:2], 2]))
            s.add_geometry(body_o_v)
        if debug_obj_o is not None:
            obj_o_v = create_vector(orient_vec=np.array([*debug_obj_o, 0]))
            obj_o_v.apply_translation(np.array([*position[0:2], 2]))
            s.add_geometry(obj_o_v)

        s.show()

    def rendering(self, anno_path, pkl, scene_mesh, indic):
        """ Render video
        """
        save_folder = os.path.dirname(anno_path)
        
        grab_render_motion_in_scene(
            smplx_folder=config.smplx_folder,
            save_folder=os.path.join(save_folder, 'rendering{:0>4d}'.format(indic)),
            pkl=pkl,
            scene_mesh=scene_mesh,
            auto_camera=False,
        )
        frame2video(
            path=os.path.join(save_folder, 'rendering{:0>4d}/%03d.png'.format(indic)),
            video=os.path.join(save_folder, 'motion_{:0>4d}.mp4'.format(indic)),
            start=0,
        )
        os.system('ffmpeg -i "{}" -vf "fps=10,scale=640:-1:flags=lanczos" -c:v pam -f image2pipe - | convert -delay 10 - -loop 0 -layers optimize "{}"'.format(
            os.path.join(save_folder, 'motion_{:0>4d}.mp4'.format(indic)),
            os.path.join(save_folder, 'motion_{:0>4d}.gif'.format(indic))
        ))

        os.system('rm -rf "{}"'.format(os.path.join(save_folder, 'rendering{:0>4d}/'.format(indic))))
    
    @staticmethod
    def update_invalid_motion_segments(motion: str):
        with open('./dataset/data/invalid_motion_segments.txt', 'a') as fp:
            fp.write(motion+'\n')



########################################
## Scene provider, a utility
########################################
class SceneProvider():
    def __init__(self, scannet_folder: str, sort: bool=False):
        self.scenes = glob.glob(os.path.join(scannet_folder, '*/*_00_vh_clean_2.labels.ply'))
        if sort:
            from natsort import natsorted
            self.scenes = natsorted(self.scenes)
        
        self.total = len(self.scenes)
        self.index = 0
        
    def next_scene(self):
        scene = self.scenes[self.index]
        self.index = (self.index + 1) % self.total

        return scene

########################################
## Motion provider, a utility
########################################
class MotionProvider():
    def __init__(self, motions: list) -> None:
        self.motions = motions

        self.total = len(self.motions)
        self.index = 0
    
    def next_motion(self):
        motion = self.motions[self.index]
        self.index = (self.index + 1) % self.total

        return motion

def align_motions_to_scene(args, MOTION_COUNT_EACH_SCENE=5):
    """ K motions to one scene, K - 1"""
    scene_provider = SceneProvider(config.scannet_folder, sort=args.sort_scene)
    if args.debug_motion is None:

        # motions = glob.glob(os.path.join(config.pure_motion_folder, args.action, '*/motion.pkl'))
        motions = glob.glob(os.path.join('/data/shared/done_grab/grabprocess/s3/test/', '*/motion.pkl'))
        # motions = ["grabtest/test/motion.pkl"]
        print(len(motions))
    else:
        motions = [args.debug_motion]
    motion_provider = MotionProvider(motions)

    alignment = Alignment(config.scan2cad_anno, config.referit3d_sr3d)
    while True:
        scene = scene_provider.next_scene()
        if scene_provider.index == 0: # end
            break

        ## each scene try to align MOTION_COUNT_EACH_SCENE motions
        for i in range(MOTION_COUNT_EACH_SCENE):
            motion = motion_provider.next_motion()

            scene_id = scene.split('/')[-2]
            motion_id = motion.split('/')[-2]
            print('Generate interaction:', scene_id, motion_id)
            print('Scene index {}, motion index {}'.format(scene_provider.index, motion_provider.index))

            anno_path = os.path.join("/data/shared/done_grab/align/s3/", scene_id[0:-2] + motion_id, 'anno.pkl')
            
            # anno_path = "./grabtest/test/anno.pkl"
            print(anno_path)

            if os.path.exists(anno_path):
                print('scene {} - motion {} had beed generated in {}'.format(scene_id, motion_id, anno_path))
                continue
            else:
                res = alignment.align_motion_portal(
                    scene, 
                    motion, 
                    args.action, 
                    anno_path, 
                    max_s=args.max_s, 
                    save_res=args.save, 
                    visualize=args.visualize, 
                    rendering=args.rendering, 
                    use_lang=args.use_lang
                )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_path', type=str)
    parser.add_argument('--action', type=str)
    parser.add_argument('--debug_motion', type=str)
    parser.add_argument('--max_s', type=int, default=2)
    parser.add_argument('--use_lang', action="store_true")
    parser.add_argument('--sort_scene', action="store_true")
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--rendering', action="store_true")
    args = parser.parse_args()
    print(args)

    align_motions_to_scene(args, 10)
