IMAGE_ENCODE = 'gpu'

import socket
import threading
import time
import re
import ast
import numpy as np
import cv2 as cv
import pickle
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation

from net import net_init, recv, send, is_connected, close
from nvjpeg import NvJpeg
nj = NvJpeg()


# ---------- Decoding helpers ----------
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


def parse_tensor_string(s: str):
    # Compatible with strings like: tensor([[...]], device='cuda:0', dtype=torch.float32)
    s = re.sub(r",?\s*device='[^']*'", '', s)
    s = re.sub(r",?\s*dtype=torch\.\w+", '', s)
    tensor_strs = re.findall(r'tensor\((\[.*?\]|\(.*?\))\)', s, re.DOTALL)
    out = []
    for ts in tensor_strs:
        try:
            data = ast.literal_eval(ts)
            out.append(np.array(data, dtype=np.float32))
        except Exception:
            pass
    return out


# ---------- Orbit camera (copied from net_viewer) ----------
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
        res = np.eye(4)
        res[2, 3] -= self.radius
        rot = np.eye(4); rot[:3, :3] = self.rot
        res = rot @ res
        res[:3, 3] += self.center
        return np.linalg.inv(res).astype(np.float32)

    @property
    def intrinsic(self):
        K = np.zeros((3, 3), dtype=np.float32)
        K[0, 0] = self.focal; K[1, 1] = self.focal; K[2, 2] = 1
        K[0, 2], K[1, 2] = self.W/2, self.H/2
        return K

    def gaussian_cam_info(self):
        return dict(w2c=self.world_to_cam, K=self.intrinsic, fovx=self.fovx/np.pi*180, height=self.H, width=self.W)

    def load_cam_pose(self, w2c):
        c2w = np.linalg.inv(w2c)
        pos = c2w[:3,3]
        self.rot = c2w[:3,:3]
        self.old_rot = self.rot
        self.radius = np.linalg.norm(pos - np.array([0,0,1]))
        self.center = pos - self.rot @ np.array([0,0,-self.radius], dtype=np.float32)
        self.old_center = self.center

    def update_orbit(self): self.old_rot = self.rot
    def orbit(self, dx, dy):
        rotvec_x = self.old_rot[:, 1] * np.radians(0.2 * dx)
        rotvec_y = self.old_rot[:, 0] * np.radians(-0.2 * dy)
        self.rot = Rotation.from_rotvec(rotvec_y).as_matrix() @ Rotation.from_rotvec(rotvec_x).as_matrix() @ self.old_rot
    def scale(self, delta): self.radius *= 1.1 ** (-delta)
    def update_pan(self): self.old_center = self.center
    def pan(self, dx, dy, dz=0): self.center = self.old_center - 2e-3 * self.rot @ np.array([dx, dy, dz])


# ---------- UDP SMPL receiver ----------
class SMPLReceiver:
    def __init__(self, host: str, port: int, on_frame):
        self.host = host
        self.port = port
        self.on_frame = on_frame
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._th.start()

    def stop(self):
        self._stop = True

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        print(f"[SMPL] Listening on {self.host or '0.0.0.0'}:{self.port}...")
        while not self._stop:
            try:
                data, _ = sock.recvfrom(65535)
                tensors = parse_tensor_string(data.decode('utf-8'))
                if len(tensors) != 3:
                    print("[SMPL] invalid tensor count, expect 3 (trans, rotvec(55x3), betas(10))")
                    continue
                transl_pelvis = np.asarray(tensors[0]).reshape(-1)
                rotvec = np.asarray(tensors[1]).reshape(-1, 3)
                shape = np.asarray(tensors[2]).reshape(-1)

                # Map to 165-dim pose: [root_orient(3), body(63), jaw(3), leye(3)=0, reye(3)=0, left_hand(45), right_hand(45)]
                root_orient = rotvec[0, :3]
                body = rotvec[1:22, :3].reshape(-1)
                jaw = rotvec[-1, :3]
                eyes = np.zeros(6, dtype=np.float32)
                hands = rotvec[22:52, :3].reshape(-1)
                pose165 = np.concatenate([root_orient, body, jaw, eyes, hands], axis=0).astype(np.float32)

                frame = dict(pose=pose165, Th=transl_pelvis.astype(np.float32), Rh=np.eye(3, dtype=np.float32))
                self.on_frame(frame)
            except Exception as e:
                print(f"[SMPL] recv error: {e}")


# ---------- GUI (adapted from net_viewer) ----------
class GUI:
    def __init__(self, height, width, ip, port, smpl_host='', smpl_port=8082):
        net_init(ip, port)
        self.cam = OrbitCamera(height=height, width=width)
        self.H, self.W = height, width
        self.image = np.zeros((height, width, 3), dtype=np.float32)
        self.timer = 0.0
        self.scaling_modifier = 1.0
        self.render_type = 'image'
        self.pose = np.zeros(165, dtype=np.float32)
        self.Th = np.array([0,0,1.1], dtype=np.float32)
        self.Rh = Rotation.from_euler('x', np.pi/2).as_matrix()
        self.background = np.ones(3, dtype=np.float32)
        self.is_test = True
        self.last_smpl_ts = 0.0

        # start UDP receiver
        self._rx = SMPLReceiver(smpl_host, smpl_port, self._on_smpl_frame)
        self._rx.start()

    def _on_smpl_frame(self, frame):
        self.pose = frame['pose']
        self.Th = frame['Th']
        self.Rh = frame['Rh']
        self.last_smpl_ts = time.time()

    def gaussian_gui_info(self):
        info = self.cam.gaussian_cam_info()
        info['scaling_modifier'] = self.scaling_modifier
        info['render_type'] = self.render_type
        info['pose'] = self.pose
        info['Th'] = self.Th
        info['Rh'] = self.Rh
        info['background'] = self.background
        info['is_test'] = self.is_test
        return info

    def loop_function(self):
        stats = dict(byte=0, frame=0)
        if not is_connected():
            return stats
        send(pickle.dumps(self.gaussian_gui_info()))
        data = recv()
        if data is None:
            return stats
        stats['byte'] = len(data)
        ret = pickle.loads(data)
        image_bytes = ret['image_bytes']
        img = decode_bytes(image_bytes)
        img = np.ascontiguousarray(img)
        dpg.set_value("texture", img)
        dpg.set_value('gaussian_num', f"Gaussian number: {ret.get('gaussian_num', '-')}")
        stats['frame'] = 1
        return stats

    # --- DPG scaffolding (minimal) ---
    def register_dpg(self):
        H, W = self.H, self.W
        dpg.create_context()
        dpg.create_viewport(title="Net Viewer (SMPL Live)", width=W+380, height=H+40)

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=W, height=H, format=dpg.mvFormat_Float_rgb, default_value=self.image, tag="texture")

        def on_reconnect():
            close(); net_init(dpg.get_value('ip'), dpg.get_value('port'))

        def on_fov():
            self.cam.update_cam(fovx=dpg.get_value('fovx')/180*np.pi)

        def on_scale(sender, app_data):
            self.scaling_modifier = app_data

        def on_bg(sender, app_data):
            bg = np.array(app_data[:3], dtype=np.float32); self.background = np.round(bg*255)/255

        with dpg.window(tag='primary_window'):
            dpg.add_image('texture', tag='image')
            dpg.set_primary_window('primary_window', True)

        with dpg.window(label='Info', width=360, pos=[W+10, 0]):
            dpg.add_text('', tag='fps')
            dpg.add_text('Gaussian number: -', tag='gaussian_num')
            dpg.add_input_text(label='ip', tag='ip', default_value='127.0.0.1', width=120)
            dpg.add_input_int(label='port', tag='port', default_value=23456, width=120)
            dpg.add_button(label='Reconnect', callback=on_reconnect)
            dpg.add_input_float(label='Fovx', tag='fovx', default_value=40, min_value=10, max_value=170, width=150)
            dpg.add_button(label='Update FOV', callback=on_fov)
            dpg.add_slider_float(default_value=1.0, min_value=0.0, max_value=1.5, label='Scaling', callback=on_scale)
            dpg.add_color_edit((255,255,255,255), label='Background', width=200, tag='background', callback=on_bg, no_alpha=True)
            dpg.add_text('SMPL last ts: -', tag='smpl_ts')

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=lambda s,a: self.cam.orbit(a[1], a[2]))
            dpg.add_mouse_wheel_handler(callback=lambda s,a: self.cam.scale(a))
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=lambda s,a: self.cam.pan(a[1], a[2]))

        dpg.setup_dearpygui(); dpg.show_viewport()

    def render_loop(self):
        dpg.set_viewport_vsync(False)
        frame_cnt, acc_t = 0, 0.0
        while dpg.is_dearpygui_running():
            stats = self.loop_function()
            dt = dpg.get_delta_time(); acc_t += dt; frame_cnt += stats['frame']
            if acc_t > 1:
                dpg.set_value('fps', f"FPS: {frame_cnt/acc_t:.1f}"); frame_cnt = 0; acc_t = 0
            if self.last_smpl_ts > 0:
                dpg.set_value('smpl_ts', f"SMPL last ts: {time.strftime('%H:%M:%S', time.localtime(self.last_smpl_ts))}")
            dpg.render_dearpygui_frame()
        dpg.destroy_context(); close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip', type=str, default='127.0.0.1')
    ap.add_argument('--port', type=int, default=23456)
    ap.add_argument('--width', type=int, default=800)
    ap.add_argument('--height', type=int, default=800)
    ap.add_argument('--smpl_host', type=str, default='')
    ap.add_argument('--smpl_port', type=int, default=8082)
    args = ap.parse_args()

    gui = GUI(height=args.height, width=args.width, ip=args.ip, port=args.port, smpl_host=args.smpl_host, smpl_port=args.smpl_port)
    gui.register_dpg(); gui.render_loop()


if __name__ == '__main__':
    main()

