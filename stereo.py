import cv2, glob, re, time, os
import numpy as np

class StereoCamera(object):
    g_sgbm = None
    g_disp_max = 256

    def __init__(self):
        self.init_sgbm()

    def __del__(self):
        print("StereoCamera exit")

    def init_sgbm(self):
        self.g_sgbm = cv2.StereoSGBM_create(
            # Minimum possible disparity value. Normally, it is zero.
            minDisparity = 0,
            # Maximum disparity minus minimum disparity. 
            # The value is always greater than zero. 
            # In the current implementation, this parameter must be divisible by 16
            numDisparities = self.g_disp_max,
            # setting blockSize=1 reduces the blocks to single pixels.
            blockSize = 1,
            # The first parameter controlling the disparity smoothness.
            P1 = 0,
            # The second parameter controlling the disparity smoothness.
            P2 = 0,
            # Maximum allowed difference (in integer pixel units) in the left-right disparity check. 
            # Set it to a non-positive value to disable the check.
            disp12MaxDiff = 0,
            # Margin in percentage by which the best (minimum) computed cost function value 
            # should "win" the second best value to consider the found match correct. 
            # Normally, a value within the 5-15 range is good enough.
            uniquenessRatio = 15,
            # Maximum size of smooth disparity regions to consider their noise speckles and invalidate. 
            # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
            speckleWindowSize = 0,
            # Maximum disparity variation within each connected component. 
            speckleRange = 0,
            # Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming algorithm. 
            mode = cv2.STEREO_SGBM_MODE_SGBM
        )
        print("sgbm init, model: {}".format(self.g_sgbm.getMode()))

    def load_pfm(self, file_path):
        """
        load image in PFM type.
        Args:
            file_path string: file path(absolute)
        Returns:
            data (numpy.array): data of image in (Height, Width[, 3]) layout
            scale (float): scale of image
        """
        with open(file_path, encoding="ISO-8859-1") as fp:
            color = None
            width = None
            height = None
            scale = None
            endian = None

            # load file header and grab channels, if is 'PF' 3 channels else 1 channel(gray scale)
            header = fp.readline().rstrip()
            if header == 'PF':
                color = True
            elif header == 'Pf':
                color = False
            else:
                raise Exception('Not a PFM file.')

            # grab image dimensions
            dim_match = re.match(r'^(\d+)\s(\d+)\s$', fp.readline())
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception('Malformed PFM header.')

            # grab image scale
            scale = float(fp.readline().rstrip())
            if scale < 0:  # little-endian
                endian = '<'
                scale = -scale
            else:
                endian = '>'  # big-endian

            # grab image data
            data = np.fromfile(fp, endian + 'f')
            shape = (height, width, 3) if color else (height, width)

            # reshape data to [Height, Width, Channels]
            data = np.reshape(data, shape)
            data = np.flipud(data)

            return data, scale

    # convert bgr to nv21: ffmpeg -i im0.png -pix_fmt nv21 im0.yuv
    def load_nv21(self, img_path, img_h, img_w):
        yuv_h = int(img_h*1.5)
        frame_len = int(yuv_h*img_w)
        fp = open(img_path, "rb")
        raw = fp.read(frame_len)
        yuv = np.frombuffer(raw, dtype=np.uint8)
        yuv = yuv.reshape(yuv_h, img_w)
        print("read yuv image {}, shape: {}".format(img_path, yuv.shape))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        return bgr

    def preview_nv21(self, img_path, img_h, img_w):
        bgr = self.load_nv21(img_path, img_h, img_w)
        cv2.imwrite(img_path+".jpg", bgr)
        cv2.namedWindow('yuv', cv2.WINDOW_NORMAL)
        cv2.imshow('yuv', bgr)
        cv2.waitKey(-1)#-1
        cv2.destroyAllWindows()

    def make_sparse_map(self, map_path, img_h, img_w):
        img_h_modulo_16 = img_h % 16
        img_w_modulo_16 = img_w % 16
        vertex_h = 0
        vertex_w = 0
        if img_h_modulo_16 == 0:
            vertex_h = int(img_h/16)+1
        else:
            vertex_h = int(img_h/16)+2

        if img_w_modulo_16 == 0:
            vertex_w = int(img_w/16)+1
        else:
            vertex_w = int(img_w/16)+2

        # generate sparse map
        fp = open(map_path, "w")
        for i in range(vertex_h):
            y = i*16
            if y > img_h-1:
                y = img_h-1
            for j in range(vertex_w):
                x= j*16
                if x > img_w-1:
                    x=img_w-1
                fp.write("{:04x}{:04x}\n".format(y<<4, x<<4))
        fp.close()
        print("original shape: {}*{}, sparse shape: {}*{}, map_path: {}".format(img_h, img_w, vertex_h, vertex_w, map_path))

    def load_disp_gt(self, img_path):
        data, scale = self.load_pfm(img_path)
        h, w = data.shape
        inf_count = 0
        dmin = np.amin(data)
        for i in range(h):
            for j in range(w):
                # print("data[{}][{}]={}".format(i, j, data[i, j]))
                if data[i, j] == np.inf:
                    inf_count+=1
                    data[i, j] = -1.0
        print("gt inf count: {}".format(inf_count))
        print("pfm file size: {}, min: {}, max: {}".format(data.shape, dmin, np.amax(data)))
        # np.savetxt("./disp_gt.txt", data, delimiter=",", fmt="%10.6f")
        return data

    def calc_disparity(self, imgL, imgR):
        print('computing disparity start @ {}'.format(time.time()))
        disp = self.g_sgbm.compute(imgL, imgR).astype(np.float32) / 16.0
        print("computing disparity end @ {}".format(time.time()))

        h, w = disp.shape
        inf_count = 0
        for i in range(h):
            for j in range(w):
                if disp[i, j] < 0:
                    inf_count+=1
                    # disp[i, j] = 0
        print("disp shape: {}, max: {}, inf_count: {}".format(disp.shape, np.amax(disp), inf_count))
        # np.savetxt("./disp_sgbm.txt", disp, delimiter=",", fmt="%10.6f")
        # disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # cv2.imwrite("./disp.sgbm.png", disp)
        # cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
        # cv2.imshow('disp', disp)
        # cv2.waitKey(-1)#-1
        # cv2.destroyAllWindows()
        return disp

    def report_diff(self, disp_sgbm, disp_gt):
        h, w = disp_gt.shape
        valid_count = 0
        pixel3_count = 0
        pixel3_avg = 0.0
        pixel3_var = 0.0
        pixel2_count = 0
        pixel2_avg = 0.0
        pixel2_var = 0.0
        pixel1_count = 0
        pixel1_avg = 0.0
        pixel1_var = 0.0

        # fp = open("./diff.txt", "w")
        for i in range(h):
            for j in range(w):
                if disp_gt[i, j]<0 or disp_sgbm[i,j]<0 or disp_gt[i, j]>self.g_disp_max or disp_sgbm[i,j]>self.g_disp_max:
                    continue
                valid_count += 1
                diff = abs(disp_gt[i, j] - disp_sgbm[i,j])
                if diff > 3.0:
                    continue
                pixel3_count += 1
                pixel3_avg += diff
                pixel3_var += diff**2
                # fp.write("{:.3f}\n".format(diff))
                    
                if diff < 2.0:
                    pixel2_count += 1
                    pixel2_avg += diff
                    pixel2_var += diff**2                    
                if diff < 1.0:
                    pixel1_count += 1
                    pixel1_avg += diff
                    pixel1_var += diff**2                    

        # fp.close()
        pixel3_avg /= pixel3_count
        pixel3_var /= pixel3_count
        pixel2_avg /= pixel2_count
        pixel2_var /= pixel2_count
        pixel1_avg /= pixel1_count
        pixel1_var /= pixel1_count
        print("[report] total valid_count: {}".format(valid_count))
        print("[report] pixel3 count: {}, avg: {}, var: {}".format(pixel3_count, pixel3_avg, pixel3_var))
        print("[report] pixel2 count: {}, avg: {}, var: {}".format(pixel2_count, pixel2_avg, pixel2_var))
        print("[report] pixel1 count: {}, avg: {}, var: {}".format(pixel1_count, pixel1_avg, pixel1_var))
        # disp_diff = np.abs(disp_sgbm-disp_gt)
        # disp_diff = cv2.normalize(disp_diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # cv2.imwrite("./disp_diff.png", disp_diff)


def test_middlebury():
    middlebury = [
        {"imgL_path":"middlebury/Adirondack-perfect/im0.png",
        "imgR_path":"middlebury/Adirondack-perfect/im1.png",
        "gt_path": "middlebury/Adirondack-perfect/disp0.pfm"},

        {"imgL_path":"middlebury/Bicycle1-perfect/im0.png",
        "imgR_path":"middlebury/Bicycle1-perfect/im1.png",
        "gt_path": "middlebury/Bicycle1-perfect/disp0.pfm"},

        {"imgL_path":"middlebury/Flowers-perfect/im0.png",
        "imgR_path":"middlebury/Flowers-perfect/im1.png",
        "gt_path": "middlebury/Flowers-perfect/disp0.pfm"},

        # {"imgL_path":"middlebury/Jadeplant-perfect/im0.png",
        # "imgR_path":"middlebury/Jadeplant-perfect/im1.png",
        # "gt_path": "middlebury/Jadeplant-perfect/disp0.pfm"},

        # {"imgL_path":"middlebury/Motorcycle-perfect/im0.png",
        # "imgR_path":"middlebury/Motorcycle-perfect/im1.png",
        # "gt_path": "middlebury/Motorcycle-perfect/disp0.pfm"},

        {"imgL_path":"middlebury/Umbrella-perfect/im0.png",
        "imgR_path":"middlebury/Umbrella-perfect/im1.png",
        "gt_path": "middlebury/Umbrella-perfect/disp0.pfm"},
    ]

    stereo = StereoCamera()
    for pkg in middlebury:
        print("\n\n##### process package: {} #####".format(pkg["gt_path"]))
        imgL = cv2.imread(pkg["imgL_path"])
        imgR = cv2.imread(pkg["imgR_path"])
        disp_sgbm = stereo.calc_disparity(imgL, imgR)
        disp_gt = stereo.load_disp_gt(pkg["gt_path"])
        stereo.report_diff(disp_sgbm, disp_gt)

def test_sparse_middlebury():
    middlebury = [
        {"imgL_path":"middlebury/Adirondack-perfect/im0_sparse.yuv",
        "imgR_path":"middlebury/Adirondack-perfect/im1_sparse.yuv",
        "gt_path": "middlebury/Adirondack-perfect/disp0.pfm",
        "map_path": "middlebury/Adirondack-perfect/sparse_map.txt",
        "img_h":1988, "img_w":2880},

        {"imgL_path":"middlebury/Bicycle1-perfect/im0_sparse.yuv",
        "imgR_path":"middlebury/Bicycle1-perfect/im1_sparse.yuv",
        "gt_path": "middlebury/Bicycle1-perfect/disp0.pfm",
        "map_path": "middlebury/Bicycle1-perfect/sparse_map.txt",
        "img_h":2008, "img_w":2988},

        {"imgL_path":"middlebury/Flowers-perfect/im0_sparse.yuv",
        "imgR_path":"middlebury/Flowers-perfect/im1_sparse.yuv",
        "gt_path": "middlebury/Flowers-perfect/disp0.pfm",
        "map_path": "middlebury/Flowers-perfect/sparse_map.txt",
        "img_h":1980, "img_w":2880},

        # {"imgL_path":"middlebury/Jadeplant-perfect/im0_sparse.yuv",
        # "imgR_path":"middlebury/Jadeplant-perfect/im1_sparse.yuv",
        # "gt_path": "middlebury/Jadeplant-perfect/disp0.pfm",
        # "map_path": "middlebury/Jadeplant-perfect/sparse_map.txt",
        # "img_h":1988, "img_w":2632},

        # {"imgL_path":"middlebury/Motorcycle-perfect/im0_sparse.yuv",
        # "imgR_path":"middlebury/Motorcycle-perfect/im1_sparse.yuv",
        # "gt_path": "middlebury/Motorcycle-perfect/disp0.pfm",
        # "map_path": "middlebury/Motorcycle-perfect/sparse_map.txt",
        # "img_h":2000, "img_w":2964},

        {"imgL_path":"middlebury/Umbrella-perfect/im0_sparse.yuv",
        "imgR_path":"middlebury/Umbrella-perfect/im1_sparse.yuv",
        "gt_path": "middlebury/Umbrella-perfect/disp0.pfm",
        "map_path": "middlebury/Umbrella-perfect/sparse_map.txt",
        "img_h":2016, "img_w":2960},
    ]

    stereo = StereoCamera()
    for pkg in middlebury:
        print("\n\n##### process package: {} #####".format(pkg["gt_path"]))
        # stereo.make_sparse_map(pkg["map_path"], pkg["img_h"], pkg["img_w"])
        # stereo.preview_nv21(pkg["imgL_path"], pkg["img_h"], pkg["img_w"])
        # stereo.preview_nv21(pkg["imgR_path"], pkg["img_h"], pkg["img_w"])
        imgL = stereo.load_nv21(pkg["imgL_path"], pkg["img_h"], pkg["img_w"])
        imgR = stereo.load_nv21(pkg["imgR_path"], pkg["img_h"], pkg["img_w"])
        disp_sgbm = stereo.calc_disparity(imgL, imgR)
        disp_gt = stereo.load_disp_gt(pkg["gt_path"])
        stereo.report_diff(disp_sgbm, disp_gt)


if __name__ == "__main__":
    test_middlebury()
    test_sparse_middlebury()
    # stereo = StereoCamera()
    # img_path = "middlebury/Adirondack-perfect/im1_sparse.yuv"
    # img_path = "middlebury/Bicycle1-perfect/im1_sparse.yuv"
    # map_path = "middlebury/Bicycle1-perfect/sparse_map.txt"
    img_h = 2008 #1080
    img_w = 2988 #1920
    # stereo.make_sparse_map(map_path, img_h, img_w)
    # stereo.preview_nv21(img_path, img_h, img_w)
