import numpy as np
import ctypes
import cv2


def convert_number_to_register_repr(number, num_regs, num_fraction_bits=0):
    norm_number = int(number * (2 ** num_fraction_bits))
    bin_num = bin(norm_number)
    num_digits = len(bin_num) - 2
    if num_digits > num_regs * 8:
        raise ValueError("Number with {} will not fit into {} 1-byte registers".format(len(bin_num) - 2, num_regs))
    elif num_digits < num_regs * 8:
        bin_num = '0b' + '0' * (num_regs * 8 - num_digits) + bin_num[2:]

    register_repr = []
    for i in range(num_regs):
        num_str = bin_num[2:][i * 8:8 * (i + 1)]
        num = int(num_str, 2)
        register_repr.append(num)
    # print(register_repr)
    return register_repr

class OVCamera:
    def __init__(self, path_to_dll='./camera_capture_old.dll',
                 AEC=436, AGC=1.625, config_path=b'camera_config.set'):
        self.path_to_dll = path_to_dll
        self.lib = ctypes.CDLL(path_to_dll)
        print('here_1')
        self.img_shape = (400, 400)
        self.num_bytes_in_image = self.img_shape[0] * self.img_shape[1] * 2
        assert isinstance(config_path, bytes), 'Config path needs to be a byte array'
        self.config_path = config_path

        # Automatic Exposure Control
        self.AEC = AEC
        # Automatic Gain Control
        self.AGC = AGC
        print('here_2')

    def start_camera(self):
        AEC = convert_number_to_register_repr(self.AEC, num_regs=3, num_fraction_bits=4)
        AGC = convert_number_to_register_repr(self.AGC, num_regs=2, num_fraction_bits=4)
        camera_controls = AEC + AGC
        c_config_path = ctypes.create_string_buffer(self.config_path)
        ret_code = self.lib.init(c_config_path, *camera_controls)
        self.success = True if ret_code == 0 else False
        if not self.success:
            print('Could not open camera succesfully')
        else:
            # Set up the return type for get_img
            self.lib.get_img_from_char_array.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_char,
                                                                              shape=(self.num_bytes_in_image,))
            self.dt = np.dtype('>u2')
        return self.success

    def close_camera(self):
        self.lib.free_img()

    def __del__(self):
        try:
            if self.success:
                self.close_camera()
        except AttributeError as err:
            print(err)

    def capture_frame(self):
        frame = self.lib.get_img_from_char_array()
        frame = np.frombuffer(frame, dtype=self.dt).astype(np.uint16)
        frame = (frame - frame.min()) / frame.max() * 255
        frame = frame.astype(np.uint8)
        frame.shape = self.img_shape
        return frame

    def write_reg_value(self, AEC, AGC):
        aec_val = AEC
        AEC = convert_number_to_register_repr(AEC, num_regs=3, num_fraction_bits=4)
        AGC = convert_number_to_register_repr(AGC, num_regs=2, num_fraction_bits=4)
        if self.path_to_dll == './camera_capture.dll':
            time_ms = 0.019 * aec_val
            fps = max(1000 / time_ms, 120)
            vts_val = 52560 / (fps - 3)
            VTS = convert_number_to_register_repr(vts_val, num_regs=2, num_fraction_bits=4)
            camera_controls = AEC + AGC + VTS
        else:
            camera_controls = AEC + AGC
        self.lib.write_reg(*camera_controls)

if __name__ == '__main__':
    cam = OVCamera()
    success = cam.start_camera()
    e = 420
    g = 3
    cam.write_reg_value(AEC=e, AGC=g)

    while True:
        img = cam.capture_frame()
        cv2.imshow('c', img)
        cv2.waitKey(10)
