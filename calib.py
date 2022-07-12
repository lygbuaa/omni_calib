import cv2, glob
import numpy as np

class OmniCalib(object):
    G_IMG_PATH = "./images/"
    G_IMG_EXT = "jpg"
    G_IMG_SIZE = (1280, 800)
    G_CHESS_SIZE = (7, 7)
    G_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    G_OMNI_FLAGS = cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER + cv2.omnidir.CALIB_FIX_XI
    G_CHESS_SQUARE_LEN = 1
    g_objp = None
    g_objpoints = []
    g_imgpoints = []
    g_images = []
    g_K = np.zeros((3,3))
    g_xi = np.array([])
    g_D = np.zeros([1,4])

    def __init__(self):
        print("opencv version: {}".format(cv2.__version__))
        self.g_objp = np.zeros((self.G_CHESS_SIZE[0] * self.G_CHESS_SIZE[1], 3), np.float32)
        self.g_objp[:, :2] = np.mgrid[0:self.G_CHESS_SIZE[0], 0:self.G_CHESS_SIZE[1]].T.reshape(-1, 2)
        self.g_objp *= self.G_CHESS_SQUARE_LEN

    def __del__(self):
        print("OmniCalib exit")

    def load_images(self):
        inputs = self.G_IMG_PATH + "*." + self.G_IMG_EXT
        self.g_images = sorted(glob.glob(inputs))
        print("found {} images: {}".format(len(self.g_images), self.g_images))

    def find_points(self):
        for fname in self.g_images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.G_CHESS_SIZE, None)
            # save corners
            if ret:
                cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), self.G_CRITERIA)
                self.g_objpoints.append(self.g_objp)
                self.g_imgpoints.append(corners)
                # show corners
                cv2.drawChessboardCorners(img, self.G_CHESS_SIZE, corners, ret)
                cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
                cv2.imshow('findCorners', img)
                cv2.waitKey(200)#-1
        cv2.destroyAllWindows()

    def omni_calib(self):
        used_img_idx = np.array([])
        images_no = len(self.g_imgpoints)
        points_no = len(self.g_imgpoints[0])
        objectpoints = np.array(self.g_objpoints, dtype=np.float64).reshape(images_no, 1, points_no, 3)
        imagepoints = np.array(self.g_imgpoints,dtype=np.float64).reshape(images_no, 1, points_no, 2)
        print("objectpoints: {}, imagepoints: {}".format(objectpoints.shape, imagepoints.shape))
        rms, self.g_K, self.g_xi, self.g_D, rvecs, tvecs, used_img_idx = cv2.omnidir.calibrate(objectpoints, imagepoints, self.G_IMG_SIZE , self.g_K, self.g_xi, self.g_D, self.G_OMNI_FLAGS, self.G_CRITERIA)
        print("omni params, K: {}, xi: {}, D: {}, used_img_idx: {}, rms error: {}".format(self.g_K, self.g_xi, self.g_D, used_img_idx, rms))

    def undistort_image(self):
        # cv2.omnidir.RECTIFY_PERSPECTIVE 
        # cv2.omnidir.RECTIFY_CYLINDRICAL
        # cv2.omnidir.RECTIFY_LONGLATI
        # cv2.omnidir.RECTIFY_STEREOGRAPHIC
        flags = cv2.omnidir.RECTIFY_PERSPECTIVE
        new_K = self.g_K
        # new_K[0][0] = self.g_K[0][0]*0.5
        # new_K[1][1] = self.g_K[1][1]*0.5
        new_size = (self.G_IMG_SIZE[0]*1, self.G_IMG_SIZE[1]*1)
        # xi=0, no distortion; xi=1, huge distortion
        self.g_xi = self.g_xi

        for fname in self.g_images:
            input_img = cv2.imread(fname)
            output_img = np.array([])
            output_img = cv2.omnidir.undistortImage(input_img, self.g_K, self.g_D, self.g_xi, flags, output_img, new_K, new_size)
            cv2.namedWindow('undistort', cv2.WINDOW_NORMAL)
            cv2.imshow('undistort', output_img)
            cv2.waitKey(-1)#-1
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    calibrator = OmniCalib()
    calibrator.load_images()
    calibrator.find_points()
    calibrator.omni_calib()
    calibrator.undistort_image()

