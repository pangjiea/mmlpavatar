IMAGE_ENCODE = 'gpu'

import socket
import threading
import time
import re
import ast
import numpy as np
import cv2 as cv
import pickle
import os
import json
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


# ---------- Orbit camera ----------
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

    def update_orbit(self): self.old_rot = self.rot
    def orbit(self, dx, dy):
        rotvec_x = self.old_rot[:, 1] * np.radians(0.2 * dx)
        rotvec_y = self.old_rot[:, 0] * np.radians(-0.2 * dy)
        self.rot = Rotation.from_rotvec(rotvec_y).as_matrix() @ Rotation.from_rotvec(rotvec_x).as_matrix() @ self.old_rot
    def scale(self, delta): self.radius *= 1.1 ** (-delta)
    def update_pan(self): self.old_center = self.center
    def pan(self, dx, dy, dz=0): self.center = self.old_center - 2e-3 * self.rot @ np.array([dx, dy, dz])


# ---------- UDP SMPL-X receiver (multi-person) ----------
class SMPLReceiverMulti:
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

    def _map_rotvec_to_pose165(self, rotvec: np.ndarray) -> np.ndarray:
        """
        Support 53-joint SMPL-X (no eyes) and 55-joint (with eyes).
        - Expected order (common 53): [0 root] [1..21 body] [22..36 lhand] [37..51 rhand] [52 jaw]
        - For 55, eyes occupy the last two indices; we still take jaw at index 52.
        """
        rotvec = rotvec.reshape(-1, 3)
        J = rotvec.shape[0]
        # Root orient
        root_orient = rotvec[0, :3]
        # Body 21 joints after root
        body = rotvec[1:22, :3].reshape(-1)  # 21*3
        # Jaw index: 52 for both 53 and 55 topologies
        jaw = rotvec[-1, :3]
        #jaw = np.zeros(3, dtype=np.float32)
        # Eyes are not provided by 53; set zeros
        # Hands (30 joints): indices 22..51
        hands = rotvec[22:52, :3].reshape(-1)
        eyes = np.zeros(6, dtype=np.float32)
        #hands = np.zeros(90, dtype=np.float32)  # 不使用手部
        pose165 = np.concatenate([root_orient, body, jaw, eyes, hands], axis=0).astype(np.float32)
        return pose165

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        print(f"[SMPLX-Multi] Listening on {self.host or '0.0.0.0'}:{self.port}...")
        while not self._stop:
            try:
                data, _ = sock.recvfrom(65535)
                arrs = parse_tensor_string(data.decode('utf-8'))
                if len(arrs) < 3:
                    continue
                # Group every 3 tensors as one person: [transl, rotvec, betas]
                people = []
                num_groups = len(arrs) // 3
                for i in range(num_groups):
                    transl_pelvis = np.asarray(arrs[3*i + 0]).reshape(-1)
                    rotvec = np.asarray(arrs[3*i + 1]).reshape(-1, 3)
                    shape = np.asarray(arrs[3*i + 2]).reshape(-1)
                    pose165 = self._map_rotvec_to_pose165(rotvec)
                    people.append(dict(
                        pose=pose165,
                        Th=transl_pelvis.astype(np.float32),
                        Rh=np.eye(3, dtype=np.float32),
                        beta=shape.astype(np.float32),
                    ))
                if people:
                    self.on_frame(dict(people=people))
            except Exception as e:
                print(f"[SMPLX-Multi] recv error: {e}")


# ---------- GUI (multi-person with selector) ----------
class GUI:
    def __init__(self, height, width, ip, port, smpl_host='', smpl_port=8083):
        net_init(ip, port)
        self.cam = OrbitCamera(height=height, width=width)
        self.H, self.W = height, width
        self.image = np.zeros((height, width, 3), dtype=np.float32)
        self.timer = 0.0
        self.scaling_modifier = 1.0
        self.render_type = 'image'

        # Multi-person state
        self.people = []  # list of dicts with keys: pose, Th, Rh
        self.person_idx = 0
        self.last_smpl_ts = 0.0
        self.background = np.ones(3, dtype=np.float32)
        self.is_test = True

        # start UDP receiver
        self._rx = SMPLReceiverMulti(smpl_host, smpl_port, self._on_smpl_frame)
        self._rx.start()

        # default save dir
        self.default_save_dir = os.path.join('output', 'snapshots')

    def _save_current_frame(self, save_dir: str = None):
        os.makedirs(save_dir or self.default_save_dir, exist_ok=True)
        save_dir = save_dir or self.default_save_dir
        ts = time.strftime('%Y%m%d_%H%M%S')
        pid = self.person_idx
        base = f'frame_{ts}_p{pid:02d}'

        # gather cam
        cam_info = self.cam.gaussian_cam_info()
        cam_to_save = dict(
            w2c=cam_info['w2c'].tolist(),
            fovx=float(cam_info['fovx']),
            height=int(cam_info['height']),
            width=int(cam_info['width']),
        )

        # gather person
        person = self._current_person()
        beta = None
        if self.people and 0 <= pid < len(self.people):
            beta = self.people[pid].get('beta', None)

        cam_path = os.path.join(save_dir, f'{base}_cam.json')
        smpl_path = os.path.join(save_dir, f'{base}_smplx.npz')
        with open(cam_path, 'w') as f:
            json.dump(cam_to_save, f, indent=2)
        if beta is None:
            np.savez(smpl_path, pose=person['pose'], Th=person['Th'], Rh=person['Rh'])
        else:
            np.savez(smpl_path, pose=person['pose'], Th=person['Th'], Rh=person['Rh'], beta=beta)
        print(f"[Save] Wrote cam: {cam_path}\n[Save] Wrote SMPL-X: {smpl_path}")

    def _on_smpl_frame(self, frame):
        self.people = frame.get('people', [])
        if self.people:
            self.person_idx = max(0, min(self.person_idx, len(self.people)-1))
        self.last_smpl_ts = time.time()

    def _current_person(self):
        if not self.people:
            # Default zeros if nothing received yet
            return dict(pose=np.zeros(165, dtype=np.float32),
                        Th=np.array([0,0,1.1], dtype=np.float32),
                        Rh=Rotation.from_euler('x', np.pi/2).as_matrix().astype(np.float32))
        return self.people[self.person_idx]

    def gaussian_gui_info(self):
        info = self.cam.gaussian_cam_info()
        p = self._current_person()
        info['scaling_modifier'] = self.scaling_modifier
        info['render_type'] = self.render_type
        info['pose'] = p['pose']
        info['Th'] = p['Th']
        info['Rh'] = p['Rh']
        info['background'] = self.background
        info['is_test'] = self.is_test
        return info

    def _drain_latest(self, max_reads: int = 5):
        latest = None
        for _ in range(max_reads):
            data = recv()
            if data is None:
                break
            latest = data
        return latest

    def loop_function(self):
        stats = dict(byte=0, frame=0)
        if not is_connected():
            return stats
        send(pickle.dumps(self.gaussian_gui_info()))
        data = recv()
        if data is None:
            for _ in range(3):
                time.sleep(0.001)
                data = recv()
                if data is not None:
                    break
        extra = self._drain_latest()
        if extra is not None:
            data = extra
        if data is None:
            return stats
        stats['byte'] = len(data)
        ret = pickle.loads(data)
        image_bytes = ret['image_bytes']
        try:
            img = decode_bytes(image_bytes)
            if img.shape[0] != self.H or img.shape[1] != self.W:
                pad_h = max(0, self.H - img.shape[0])
                pad_w = max(0, self.W - img.shape[1])
                img = np.pad(img, [[0, pad_h], [0, pad_w], [0, 0]])
                img = img[: self.H, : self.W]
            img = np.ascontiguousarray(img)
        except Exception:
            # Fallback path if decode fails (already float32 HxWx3)
            img = image_bytes
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255
        dpg.set_value("texture", img)
        dpg.set_value('gaussian_num', f"Gaussian number: {ret.get('gaussian_num', '-')}")
        stats['frame'] = 1
        return stats

    # --- DPG scaffolding (minimal) ---
    def register_dpg(self):
        H, W = self.H, self.W
        dpg.create_context()
        dpg.create_viewport(title="Net Viewer (SMPL-X Multi)", width=W+420, height=H+40)

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

        def on_pick_person(sender, app_data):
            self.person_idx = int(app_data)

        with dpg.window(tag='primary_window'):
            dpg.add_image('texture', tag='image')
            dpg.set_primary_window('primary_window', True)

        with dpg.window(label='Info', width=380, pos=[W+10, 0]):
            dpg.add_text('', tag='fps')
            dpg.add_text('Gaussian number: -', tag='gaussian_num')
            dpg.add_text('People: 0', tag='people_cnt')
            dpg.add_input_text(label='ip', tag='ip', default_value='127.0.0.1', width=120)
            dpg.add_input_int(label='port', tag='port', default_value=23456, width=120)
            dpg.add_button(label='Reconnect', callback=on_reconnect)
            dpg.add_input_float(label='Fovx', tag='fovx', default_value=40, min_value=10, max_value=170, width=150)
            dpg.add_button(label='Update FOV', callback=on_fov)
            dpg.add_slider_float(default_value=1.0, min_value=0.0, max_value=1.5, label='Scaling', callback=on_scale)
            dpg.add_color_edit((255,255,255,255), label='Background', width=200, tag='background', callback=on_bg, no_alpha=True)
            dpg.add_text('SMPL last ts: -', tag='smpl_ts')
            dpg.add_slider_int(label='Person Idx', tag='person_idx', default_value=0, min_value=0, max_value=0, callback=on_pick_person)
            # Save controls
            dpg.add_separator()
            dpg.add_input_text(label='Save Dir', tag='save_dir', default_value=self.default_save_dir, width=260)
            dpg.add_button(label='Save Current (S)', callback=lambda: self._save_current_frame(dpg.get_value('save_dir')))

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=lambda s,a: self.cam.orbit(a[1], a[2]))
            dpg.add_mouse_wheel_handler(callback=lambda s,a: self.cam.scale(a))
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=lambda s,a: self.cam.pan(a[1], a[2]))
            # hotkey: S to save current snapshot
            dpg.add_key_press_handler(key=dpg.mvKey_S, callback=lambda: self._save_current_frame(dpg.get_value('save_dir')))

        dpg.setup_dearpygui(); dpg.show_viewport()

    def render_loop(self):
        dpg.set_viewport_vsync(True)
        frame_cnt, acc_t = 0, 0.0
        while dpg.is_dearpygui_running():
            stats = self.loop_function()
            dt = dpg.get_delta_time(); acc_t += dt; frame_cnt += stats['frame']
            if acc_t > 1:
                dpg.set_value('fps', f"FPS: {frame_cnt/acc_t:.1f}"); frame_cnt = 0; acc_t = 0
            # Update multi-person UI state
            n = len(self.people)
            dpg.set_value('people_cnt', f"People: {n}")
            dpg.configure_item('person_idx', max_value=max(0, n-1))
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
    ap.add_argument('--smpl_host', type=str, default='')  # bind all by default
    ap.add_argument('--smpl_port', type=int, default=8082)  # match multi-hmr_v6 sender default
    args = ap.parse_args()

    gui = GUI(height=args.height, width=args.width, ip=args.ip, port=args.port, smpl_host=args.smpl_host, smpl_port=args.smpl_port)
    gui.register_dpg(); gui.render_loop()


if __name__ == '__main__':
    main()
