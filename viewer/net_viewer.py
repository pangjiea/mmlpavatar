IMAGE_ENCODE = 'gpu'

import os
from os import path
import numpy as np
import cv2 as cv
import pickle
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation
import json

from net import net_init, recv, send, is_connected, close
from nvjpeg import NvJpeg
nj = NvJpeg()

left_hand_pose = np.array([0.09001956135034561, 0.1604590266942978, -0.3295670449733734, 0.12445037066936493, -0.11897698789834976, -1.5051144361495972, -0.1194705069065094, -0.16281449794769287, -0.6292539834976196, -0.27713727951049805, 0.035170216113328934, -0.5893177390098572, -0.20759613811969757, 0.07492011040449142, -1.4485805034637451, -0.017797302454710007, -0.12478633224964142, -0.7844052314758301, -0.4157009720802307, -0.5140947103500366, -0.2961726784706116, -0.7421528100967407, -0.11505582183599472, -0.7972996830940247, -0.29345276951789856, -0.18898937106132507, -0.6230823397636414, -0.18764786422252655, -0.2696149945259094, -0.5542467832565308, -0.47717514634132385, -0.12663133442401886, -1.2747308015823364, -0.23940050601959229, -0.1586960405111313, -0.7655659914016724, 0.8745182156562805, 0.5848557353019714, -0.07204405218362808, -0.5052485466003418, 0.1797526329755783, 0.3281439244747162, 0.5276764035224915, -0.008714836090803146, -0.4373648762702942], dtype = np.float32)
right_hand_pose = np.array([0.034751810133457184, -0.12605343759059906, 0.5510415434837341, 0.19454114139080048, 0.11147838830947876, 1.4676157236099243, -0.14799435436725616, 0.17293521761894226, 0.4679432511329651, -0.3042353689670563, 0.007868679240345955, 0.8570928573608398, -0.1827319711446762, -0.07225851714611053, 1.307037591934204, -0.02989627793431282, 0.1208646297454834, 0.7142824530601501, -0.3403030335903168, 0.5368582606315613, 0.3839572072029114, -0.9722614884376526, 0.17358140647411346, 0.911861002445221, -0.29665058851242065, 0.21779759228229523, 0.7269846796989441, -0.15343312919139862, 0.3083758056163788, 0.7146623730659485, -0.5153037309646606, 0.1721675992012024, 1.2982604503631592, -0.2590428292751312, 0.12812566757202148, 0.7502076029777527, 0.8694817423820496, -0.5263001322746277, 0.06934576481580734, -0.4630220830440521, -0.19237111508846283, -0.25436165928840637, 0.5972414612770081, -0.08250168710947037, 0.5013565421104431], dtype = np.float32)
hand_pose = np.concatenate([left_hand_pose, right_hand_pose], axis=0)

class OrbitCamera:
    def __init__(self, height, width, center=np.array([0,0,0]), radius=1.5, fovx=np.pi/4):
        self.W, self.H = width, height
        self.radius = radius
        self.center = center.astype(np.float32)
        self.fovx = fovx
        self.rot = np.eye(3)

        self.focal = self.W / 2 / np.tan(self.fovx/2)
        self.fovy = 2 * np.arctan(self.H / 2 / self.focal)

        self.old_rot = self.rot
        self.old_center = self.center

    def update_cam(self, width=None, height=None, fovx=None):
        if width is not None: self.W = width
        if height is not None: self.H = height
        if fovx is not None: self.fovx = fovx

        self.focal = self.W / 2 / np.tan(self.fovx/2)
        self.fovy = 2 * np.arctan(self.H / 2 / self.focal)

    @property
    def world_to_cam(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # translate
        res[:3, 3] += self.center

        res = np.linalg.inv(res)
        return res.astype(np.float32)
    
    @property
    def intrinsic(self):
        K = np.zeros((3, 3))
        K[0, 0] = self.focal
        K[1, 1] = self.focal
        K[2, 2] = 1
        K[0, 2], K[1, 2] = self.W/2, self.H/2
        return K.astype(np.float32)

    def gaussian_cam_info(self):
        data = dict(
            w2c=self.world_to_cam,
            K=self.intrinsic,
            fovx=self.fovx / np.pi * 180,
            height=self.H,
            width=self.W,
        )
        return data
    
    def load_cam_pose(self, w2c):
        c2w = np.linalg.inv(w2c)
        pos = c2w[:3,3]
        self.rot = c2w[:3,:3]
        self.old_rot = self.rot
        self.radius = np.linalg.norm(pos - np.array([0,0,1]))
        self.center = pos - self.rot @ np.array([0,0,-self.radius], dtype=np.float32)
        self.old_center = self.center

    def update_orbit(self):
        self.old_rot = self.rot

    def orbit(self, dx, dy):
        rotvec_x = self.old_rot[:, 1] * np.radians(0.2 * dx)
        rotvec_y = self.old_rot[:, 0] * np.radians(-0.2 * dy)
        self.rot = Rotation.from_rotvec(rotvec_y).as_matrix() @ \
            Rotation.from_rotvec(rotvec_x).as_matrix() @ \
            self.old_rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def update_pan(self):
        self.old_center = self.center

    def pan(self, dx, dy, dz=0):
        self.center = self.old_center - 2e-3 * self.rot @ np.array([dx, dy, dz])


def decode_bytes(image_byte):
    if IMAGE_ENCODE == 'cpu':
        image_byte = np.frombuffer(image_byte, dtype=np.uint8)
        image = cv.imdecode(image_byte, cv.IMREAD_COLOR)
    elif IMAGE_ENCODE == 'gpu':
        image = nj.decode(image_byte)
    else:
        image = image_byte 
    image = image.astype(np.float32) / 255
    return image


def crop_image_to_shape(image, height, width):
    if image.shape[0] == height and image.shape[1] == width: return image

    pad_h = max(0, height - image.shape[0])
    pad_w = max(0, width - image.shape[1])
    image = np.pad(image, [[0,pad_h],[0,pad_w],[0,0]])
    return image[:height,:width]


def load_amass_pose_list(pose_path):
    data = np.load(pose_path)
    pose_list = []
    poses = data['poses'].astype(np.float32)
    trans = data['trans'].astype(np.float32)
    N = len(poses)

    OPTIMIZE_AMASS = True
    if OPTIMIZE_AMASS:
        foo = poses[:,3:]
        foo[:, 13 * 3 + 2] -= 0.25
        foo[:, 12 * 3 + 2] += 0.25
        foo[:, 19 * 3: 20 * 3] = 0.
        foo[:, 20 * 3: 21 * 3] = 0.
        foo[:, 14 * 3] = 0.

        poses[:,3:] = foo

        # smooth
        win_size = 1
        poses_clone = np.copy(poses)
        trans_clone = np.copy(trans)
        frame_num = poses_clone.shape[0]
        poses[win_size: frame_num-win_size] = 0
        trans[win_size: frame_num-win_size] = 0
        for i in range(-win_size, win_size + 1):
            poses[win_size: frame_num-win_size] += poses_clone[win_size+i: frame_num-win_size+i]
            trans[win_size: frame_num-win_size] += trans_clone[win_size+i: frame_num-win_size+i]
        poses[win_size: frame_num-win_size] /= (2 * win_size + 1)
        trans[win_size: frame_num-win_size] /= (2 * win_size + 1)

    for i in range(N):
        pose_list.append(dict(pose=poses[i], Th=trans[i], Rh=np.eye(3, dtype=np.float32)))

    return pose_list

def load_thuman_pose_list(pose_path):
    smpl_params = np.load(pose_path, allow_pickle=True)
    smpl_params = dict(smpl_params)

    pose_list = []
    N = len(smpl_params['global_orient'])
    for frame_id in range(N):
        pose = np.concatenate([smpl_params['global_orient'][frame_id],
                    smpl_params['body_pose'][frame_id],
                    np.zeros(3,dtype=np.float32),
                    np.zeros(6,dtype=np.float32),
                    smpl_params['left_hand_pose'][frame_id],
                    smpl_params['right_hand_pose'][frame_id],], axis=0)
        Th = smpl_params['transl'][frame_id]
        Rh = np.eye(3, dtype=np.float32)
        pose_list.append(dict(pose=pose, Th=Th, Rh=Rh))
    return pose_list

class GUI:
    def __init__(self, height, width, ip, port):
        net_init(ip, port)
        self.cam = OrbitCamera(height=height, width=width)
        self.H = height
        self.W = width
        self.image = None
        self.timer = 0

        self.scaling_modifier = 1.0
        self.camera_list = []
        self.pose_list = []
        self.render_type = 'image'

        self.pose = np.zeros(165, dtype=np.float32)
        self.Th = np.array([0,0,1.1], dtype=np.float32)
        self.Rh = Rotation.from_euler('x', np.pi/2).as_matrix()

        self.background = np.ones(3, dtype=np.float32)

        self.novel_pose_list = []

        self.image = np.zeros((height, width, 3), dtype=np.float32)

        self.edit_resize_texture = None
        self.edit_texture_position = None

        self.is_transl = True
        self.is_test = True
        self.is_fist = False

    def gaussian_gui_info(self):
        info = self.cam.gaussian_cam_info()
        def set_val(s, v):
            if v is not None: info[s] = v

        pose = np.copy(self.pose)
        if self.is_fist:
            pose[-len(hand_pose):] = hand_pose

        set_val('scaling_modifier', self.scaling_modifier)
        set_val('render_type', self.render_type)
        set_val('pose', pose)
        set_val('Th', self.Th)
        set_val('Rh', self.Rh)
        set_val('background', self.background)
        set_val('is_test', self.is_test)

        return info

    def load_camera_list(self, cameras):
        self.camera_list = cameras
        dpg.configure_item('camera_id', max_value=len(self.camera_list))

    def load_pose_list(self, poses):
        self.pose_list = poses
        dpg.configure_item('frame_id', max_value=len(self.pose_list)-1)

    def loop_function(self):
        info=dict(byte=0, frame=0)

        if not is_connected(): return info
        
        data = self.gaussian_gui_info()
        send(pickle.dumps(data))

        data = recv()
        if data is None: return info
        info['byte'] = len(data)
        data = pickle.loads(data)

        if 'camera_list' in data: self.load_camera_list(data['camera_list'])
        if 'pose_list' in data: self.load_pose_list(data['pose_list'])
        if 'gaussian_num' in data: dpg.set_value('gaussian_num', 'Gaussian number: ' + str(data['gaussian_num']))

        image_bytes = data['image_bytes']
        image = decode_bytes(image_bytes)
        self.image = crop_image_to_shape(image, self.H, self.W)
        dpg.set_value('texture', self.image)
        info['frame'] = 1
        return info

    def render_loop(self):

        frame_cnt = 0
        total_byte = 0
        acc_time = 0 

        dpg.set_viewport_vsync(False)

        while dpg.is_dearpygui_running():
            info = self.loop_function()

            elapsed_time = dpg.get_delta_time()
            acc_time += elapsed_time
            frame_cnt += info['frame']
            total_byte += info['byte']

            self.timer += elapsed_time
            dpg.set_value('timer', f'Timer: {self.timer:.2f}')
            if acc_time > 1: 
                fps = frame_cnt / acc_time
                dpg.set_value('fps', f'FPS: {fps:.1f} ')
                dpg.set_value('mbps', f'{total_byte / 10e6} MB/s')
                frame_cnt, acc_time, total_byte = 0, 0, 0

            dpg.render_dearpygui_frame()
        dpg.destroy_context()  
        close()     

    def register_dpg(self):
        cam = self.cam
        H, W = self.H, self.W
        W_info = 500

        dpg.create_context()
        dpg.create_viewport(title="Net Viewer", width=W+20+W_info, height=H+20)
        # dpg.set_viewport_pos([2100, 500])

        # register mouse callback for cameras
        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("primary_window"): return
            cam.orbit(app_data[1], app_data[2])

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("primary_window"): return
            cam.scale(app_data)

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("primary_window"): return
            cam.pan(app_data[1], app_data[2])
        
        def callback_camera_release_rotate(sender, app_data):
            if not dpg.is_item_focused("primary_window"): return   
            cam.update_orbit()     

        def callback_camera_release_pan(sender, app_data):
            if not dpg.is_item_focused("primary_window"): return   
            cam.update_pan()   

        # register info
        def callback_reconnect(sender, app_data):
            close()
            ip = dpg.get_value('ip')
            port = dpg.get_value('port')
            net_init(ip, port)

        def callback_reset_timer(sender, app_data):
            self.timer = 0

        def callback_update_camera(sender, app_data):
            fovx = dpg.get_value('fovx')
            cam.update_cam(fovx=fovx / 180 * np.pi)

        def callback_scale_slider(sender, app_data):
            self.scaling_modifier = app_data

        def callback_camera_id(sender, app_data):
            cam_info = self.camera_list[app_data]
            cam.load_cam_pose(cam_info['w2c'])

        def callback_render_type(sender, app_data):
            self.render_type = app_data

        def callback_frame_id(sender, app_data):
            pose_info = self.pose_list[app_data]
            self.pose = pose_info['pose']
            self.Rh = pose_info['Rh']
            self.Th = pose_info['Th']

        def callback_tpose(sender, app_data):
            self.pose = np.zeros(165, dtype=np.float32)
            self.Rh = Rotation.from_euler('x', 90, degrees=True).as_matrix()
            self.Th = np.array([0, 0, 1.1], dtype=np.float32)

        def callback_bigpose(sender, app_data):
            big_poses = np.zeros(165, dtype=np.float32)
            big_poses[5] = np.deg2rad(30)
            big_poses[8] = np.deg2rad(-30)   
            
            self.pose = big_poses
            self.Rh = Rotation.from_euler('x', 90, degrees=True).as_matrix()
            self.Th = np.array([0, 0, 1.1], dtype=np.float32)

        def callback_fist(sender, app_data):
            self.is_fist = app_data

        def callback_is_test(sender, app_data):
            self.is_test = app_data

        def callback_background(sender, app_data):
            background = np.array(app_data[:3], dtype=np.float32)
            background = np.round(background * 255) / 255
            self.background = background

        def callback_load_novel_pose(sender, app_data):
            pose_path = dpg.get_value('novel_pose_text')
            if not path.exists(pose_path): return
            if 'smpl_params.npz' in pose_path:
                pose_list = load_thuman_pose_list(pose_path)
            else:
                pose_list = load_amass_pose_list(pose_path)
            self.novel_pose_list = pose_list

            dpg.configure_item('novel_pose_frame_id', max_value=len(self.novel_pose_list)-1)

        def callback_novel_frame_id(sender, app_data):
            pose_info = self.novel_pose_list[app_data]
            self.pose = pose_info['pose']
            self.Rh = pose_info['Rh']
            self.Th = pose_info['Th'] 
            if not self.is_transl: 
                self.Th = np.zeros_like(pose_info['Th'])

        def callback_is_no_transl(sender, app_data):
            self.is_transl = not app_data

        def callback_load_camera(sender, app_data):
            camera_path = dpg.get_value('camera_path')
            if not path.exists(camera_path): return
            with open(camera_path, 'r') as file:
                info = json.load(file)
            
            w2c = np.array(info['w2c']).reshape(4,4).astype(np.float32)
            cam.load_cam_pose(w2c)

        def callback_save_camera(sender, app_data):
            camera_path = dpg.get_value('camera_path')
            info = self.cam.gaussian_cam_info()
            info['w2c'] = info['w2c'].reshape(-1).tolist()
            info['K'] = info['K'].reshape(-1).tolist()
            with open(camera_path, 'w') as file:
                json.dump(info, file)

        def my_separator():
            dpg.add_text('')
            dpg.add_separator()
            dpg.add_text('')

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width=W,
                height=H,
                format=dpg.mvFormat_Float_rgb,
                default_value=np.zeros((H,W,3), dtype=np.float32),
                tag="texture")

        with dpg.window(tag='primary_window'):
            dpg.add_image('texture', tag='image')
            dpg.set_primary_window('primary_window', True)

        with dpg.window(label='Info', width=W_info, pos=[W+10, 0]):

            with dpg.group(horizontal=True, xoffset=250):
                dpg.add_text('', tag='fps')
                dpg.add_text('- MB/s', tag='mbps')

            with dpg.group(horizontal=True, xoffset=250):
                dpg.add_text('Timer: -', tag='timer')
                dpg.add_button(label='Reset', callback=callback_reset_timer)

            my_separator()

            with dpg.group(horizontal=True, xoffset=250):
                dpg.add_input_text(label='ip', tag='ip', default_value='127.0.0.1', width=150)
                dpg.add_input_int(label='port', tag='port', default_value=23456, width=150)
            dpg.add_button(label='Reconnect', callback=callback_reconnect)

            with dpg.group(horizontal=True, xoffset=250):
                # dpg.add_input_int(label='Width', tag='image_width', default_value=W, width=150)
                # dpg.add_input_int(label='Height', tag='image_height', default_value=H, width=150)

                dpg.add_text(f'Width: {W}')
                dpg.add_text(f'Height: {H}')

            dpg.add_input_float(label='Fovx', tag='fovx', default_value=40, min_value=10, max_value=170, width=150)
            dpg.add_button(label='Update camera', callback=callback_update_camera)

            my_separator()

            dpg.add_text('Gaussian number: -', tag='gaussian_num')
            dpg.add_slider_float(default_value=1, min_value=0, max_value=1, label='Scaling Modifier', tag='scale_modifier', callback=callback_scale_slider)
            dpg.add_input_int(default_value=0, label='Camera id', tag='camera_id', min_value=0, max_value=0, min_clamped=True, max_clamped=True, callback=callback_camera_id)
            dpg.add_combo(['image',], label='Render Type', default_value=self.render_type, callback=callback_render_type)

            with dpg.group(horizontal=True, xoffset=100):
                dpg.add_button(label='Big pose', tag='big_pose', callback=callback_bigpose)
                dpg.add_button(label='T pose', tag='t_pose', callback=callback_tpose)
                dpg.add_checkbox(label='Fist', callback=callback_fist, default_value=False)
                dpg.add_checkbox(label='Test Mode', callback=callback_is_test, default_value=True)

            dpg.add_color_edit((255, 255, 255, 255), label="Background color", width=200, tag='background', callback=callback_background, no_alpha=True)

            my_separator()

            with dpg.group(horizontal=True, xoffset=150):
                dpg.add_text('Novel Pose Path')
                dpg.add_button(label='Load', callback=callback_load_novel_pose)
                dpg.add_checkbox(label='No Transl', callback=callback_is_no_transl)
            dpg.add_text('Load novel poses from AMASS pose file, or smpl_params.npz')
            dpg.add_input_text(tag='novel_pose_text', default_value='')

            my_separator()

            with dpg.group(horizontal=True, xoffset=150):
                dpg.add_text('Save/Load Camera')
                dpg.add_button(label='Save', callback=callback_save_camera)
                dpg.add_button(label='Load', callback=callback_load_camera)
            dpg.add_text('Save/load the camera to/from a JSON file, e.g., /tmp/camera.json')
            dpg.add_input_text(tag='camera_path', default_value='')

        with dpg.window(label='Frame', width=W_info, pos=[W+10, 700]):
            dpg.add_text('Training pose')
            dpg.add_slider_int(default_value=0, label='Frame id', tag='frame_id', min_value=0, max_value=0, callback=callback_frame_id)
            
            dpg.add_text('Novel pose')
            dpg.add_slider_int(default_value=0, label='Frame id', tag='novel_pose_frame_id', min_value=0, max_value=0, callback=callback_novel_frame_id)


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_release_rotate)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_release_pan)

        dpg.setup_dearpygui()
        dpg.show_viewport()

def main():
    H, W = 1000, 1000
    try:
        gui = GUI(height=H, width=W, ip='127.0.0.1', port=23456)
        gui.register_dpg()
        gui.render_loop()
    except KeyboardInterrupt:
        close()
        raise

if __name__ == '__main__':
    main()