"""
Stitch together a panorama from a given video [input]
"""

#TODO: remove all os filepath manipulations - savings and stuff

import os
from typing import Tuple, Union
import cv2
import numpy as np
import glob
# import time


import PySimpleGUI as sg

class Stitcher(object):
    '''
        A Stitcher object for creating panoramas from videos. Must provide GUI window.
    '''
    def __init__(self, window: sg.Window, max_match: int = 100, focal_length: int = 3200, resize_factor: int = 1):
        self.orb = cv2.ORB_create() # ORB (Oriented FAST and Rotated BRIEF) a key-point detector
                                    # and desciptor to desctibe the overlap between frames
        self.panorama_frames = []
        self.frame_dump = []

        self.__filepath = None
        self.FPS = None

        self.min_match_num = 40
        self.max_match_num = max_match

        self.__resize = resize_factor

        self.debug = False

        self.f = focal_length

        self.__pano = None

        self.window = window

# ------ Retrieve frames ------ #

    #TODO: refactor this later
    def extract_frames(self) -> None:
        """
        Extract (ideal) frames from the video stream (using cv2)
        Mainly from this paper:
        https://github.com/ybhan/Panoramic-Photo-Generation/blob/master/Panoramic-Photo-Generation.pdf
        """
        assert self.__filepath is not None, "No filepath provided"

        # print(self.max_match_num)
        # From https://github.com/ybhan/Panoramic-Photo-Generation
        # Construct VideoCapture object to get frame-by-frame stream\
        vid_cap = cv2.VideoCapture(self.__filepath)
        total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = vid_cap.get(cv2.CAP_PROP_FPS)

        # glutInit(sys.argv)

        # view = OGLVideoView(vid_cap)
        # view.main()

        # The first key frame (frame0.jpg) is selected by default
        success, last = vid_cap.read()
        assert success, "couldn't read first frame"
        # last = cylindrical_project(last)
        # last = cv2.resize(last, (640, 360))
        if self.debug:
            if not os.path.isdir('key_frames'):
                os.mkdir('key_frames')

            for file_path in glob.glob('key_frames/*.jpg'):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print("Error: %s : %s" % (file_path, e.strerror))

        # tic = time.time()
        last = self.cylindrical_project(last)
        # toc = time.time()
        # print(f"Cylindrical warp time: {toc - tic} 0/{total_frames}")

        if self.debug:
            cv2.imwrite('key_frames/frame0.jpg', last)
            # print("Captured frame0.jpg")
        frame_num = 1

        image = last # An unnecessary assignment

        # Save first frame
        self.frame_dump.append(image)
        self.panorama_frames.append(image)

        # Read second frame
        success, image = vid_cap.read()

        debug_num = 1

        while success:
            # Run a progress bar for the gui
            # Rerun value is of (subprocess number, current place, total in subsection, text)
            self.window.write_event_value('-UPDATE PROGRESS BAR-', (1, debug_num, total_frames,
                                                             f'Reading frame {debug_num}/{total_frames} to panorama'))

            # tic = time.time()
            image = self.cylindrical_project(image)
            # toc = time.time()
            # print(f"Cylindrical warp time: {toc - tic} {debug_num}/{total_frames}")
            self.frame_dump.append(image)

            success_H, _ = self.find_homography(last, image)

            if success_H:
                #TODO: Consider making Pair object with Pair.H attribute
                # Frame is good for panorama, save frame.
                self.panorama_frames.append(image)

                # Reassign last
                last = image

                if self.debug:
                    # Save key frame as JPG file
                    # print("Captured frame{}.jpg".format(frame_num))
                    cv2.imwrite('key_frames/frame%d.jpg' % frame_num, last)
                    frame_num += 1

            success, image = vid_cap.read()
            debug_num += 1
            # print(debug_num)

# ------ Build the panorama ------ #

    def stitch(self, given_input: Union[str, list[np.ndarray]]) -> np.ndarray:
        '''
        Recives either: a file path to extract panorama frames and frame dump from OR a paborama frames list.
        Runs through stitching algorithm.
        Returns panorama.
        '''
        # input is the filepath
        if isinstance(given_input, str):
            # Set the file path
            self.set_filepath(given_input)

            # Fill self.panorama_frames and self.frame_dump
            self.extract_frames()
        # input is a list of panorama frames
        elif isinstance(given_input[0], np.ndarray):
            self.panorama_frames = given_input
        # something is wrong
        else:
            raise TypeError(...)

        # Build the panorama
        self.build_panorama()

        # Rerun value is of (subprocess number, current place, total in subsection, text)
        self.window.write_event_value('-UPDATE PROGRESS BAR-', (2, 4, 4, "Done!"))

        return self.__pano

    def build_panorama(self) -> None:
        '''
        Gets list of np.ndarray frames strictly for the panorama.
        Makes the (img1, img2, H) triple-tuple and sends to stitch_frames.
        Returns final panorama.
        '''
        num_frames = len(self.panorama_frames)
        if num_frames == 1:
            self.__pano = self.panorama_frames[0]
            return
        # assert num_frames > 2, "Frame loading problem. Choose different match_nums."
        pair_tuples = [] # (img1, img2, H)

        self.set_max_match_num(np.inf)

        for i in range(num_frames - 1):
            # Find homography b.w. panorama_frames[i] and panorama_frames[i+1]
            success_H, H = self.find_homography(self.panorama_frames[i], self.panorama_frames[i+1])

            if not success_H:
                self.window.write_event_value('-UPDATE PROGRESS BAR-', (0,0,1,
                                                                        f'Resize factor ({self.__resize}) too big. Please exit program and try again.'))
                
                # assert success_H, "couldn'd build final panorama"
                return None

            pair_tuples.append((self.panorama_frames[i], self.panorama_frames[i+1], H))

        self.stitch_frames(pair_tuples) # Saves self.__pano

    def stitch_frames(self, frame_pairs: list[np.ndarray]) -> None:
        '''
        Takes (frame1, frame2, H) pairs and stitches to a panorama.
        Starts from middle and proceeds to edges.
        '''
        num_frames = len(frame_pairs) + 1 # Num of pairs + 1
        # print(f'num frames: {num_frames}')
        if num_frames == 2:
            # Rerun value is of (subprocess number, current place, total in subsection, text)
            self.window.write_event_value('-UPDATE PROGRESS BAR-', (2, 0, 4,
                                                                f"Stitching panorama's right side 1/{num_frames}"))
            right_panorama = frame_pairs[0][1]

            # Rerun value is of (subprocess number, current place, total in subsection, text)
            self.window.write_event_value('-UPDATE PROGRESS BAR-', (2, 1, 4,
                                                                f"Stitching panorama's right side 2/{num_frames}"))
            left_panorama = frame_pairs[0][0]
        else:
            i = int(num_frames / 2) # Middle of list

            # Rerun value is of (subprocess number, current place, total in subsection, text)
            self.window.write_event_value('-UPDATE PROGRESS BAR-', (2, 0, 4,
                                                                f"Stitching panorama's right side {num_frames - i}/{num_frames}"))
            right_panorama = right_stitch(frame_pairs[i:])

            # Rerun value is of (subprocess number, current place, total in subsection, text)
            self.window.write_event_value('-UPDATE PROGRESS BAR-', (2, 1, 4,
                                                                f"Stitching panorama's left side {num_frames}/{num_frames}"))
            left_panorama = left_stitch(frame_pairs[:i])

        # Rerun value is of (subprocess number, current place, total in subsection, text)
        self.window.write_event_value('-UPDATE PROGRESS BAR-', (2, 2, 4,
                                                            "Stitching left side and right side together"))

        # print("Merging both sides")
        self.set_max_match_num(np.inf)
        success_H, final_H = self.find_homography(left_panorama, right_panorama)
        
        if not success_H:
            self.window.write_event_value('-UPDATE PROGRESS BAR-', (0,0,1,
                                                                    f'f ({self.f}) too small. Please exit program and try again.'))
            
            # assert success_H, "couldn'd build final panorama"
            return None
        
        final_H = np.linalg.inv(final_H)
        offset, new_size = offset_and_size(left_panorama, right_panorama, final_H)
        final_H = offset @ final_H

        rightImg_top_left, rightImg_btm_right = get_corners(right_panorama, final_H) #(x, y)

        right_panorama = cv2.warpPerspective(right_panorama, final_H, new_size)

        leftImg_top_left, leftImg_btm_right = get_corners(left_panorama, offset) #(x, y)

        left_panorama = cv2.warpPerspective(left_panorama, offset, new_size)

        depth = left_panorama.shape[2]

        assert right_panorama.shape[2] == depth, 'dimention mismatch'

        panorama = np.where(
            np.repeat(np.sum(left_panorama, axis=2)[:, :, np.newaxis], depth, axis=2) != 0,
            left_panorama,
            right_panorama,
        ).astype(np.uint8)

        top_y = min(leftImg_top_left[1], rightImg_top_left[1])
        btm_y = max(leftImg_btm_right[1], rightImg_btm_right[1])
        left_x = min(leftImg_top_left[0], rightImg_top_left[0])
        right_x = max(leftImg_btm_right[0], rightImg_btm_right[0])

        panorama = panorama[top_y:btm_y, left_x:right_x]

        if self.debug:
            cv2.imwrite('key_frames/panorama.jpg', panorama)

        self.__pano = panorama

    def cylindrical_project(self, img: np.ndarray) -> np.ndarray:
        """Inverse interpolation is applied to find the cylindrical projection of
        the original image.
        From https://www.morethantechnical.com/2018/10/30/cylindrical-image-warping-for-panorama-stitching/
        """
        height, width = img.shape[:2]
        height = int(height / self.__resize)
        width = int(width / self.__resize)
        img = cv2.resize(img, (width, height))

        centerX = int(width / 2)
        centerY = int(height / 2)

        K = np.array([[self.f, 0, centerX],
                    [0, self.f, centerY],
                    [0, 0, 1]], dtype=np.int32)
        # print(np.linalg.det(K))

        y_i, x_i = np.indices((height,width))

        X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(height*width,3)

        Kinv = np.linalg.inv(K) 

        X = (Kinv @ X.T).T # normalized coords

        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(width*height,3)

        B = (K @ A.T).T # project back to image-pixels plane

        # back from homog coords
        B = B[:,:-1] / B[:,[-1]]

        # make sure warp coords only within image bounds
        B[(B[:,0] < 0) | (B[:,0] >= width) | (B[:,1] < 0) | (B[:,1] >= height)] = -1

        B = B.reshape(height,width,-1)
        
        img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...

        # warp the image according to cylindrical coords
        final_img = cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

        return final_img

    def find_homography(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[bool, np.ndarray]:
        '''
        Find homography from two images.
        Takes the two images, a key-poind descriptor (defult is orb). Default min and max mask match given.
        Return success value, Homography matrix.
        '''
        # Detect and compute key points and descriptors:
        # https://medium.com/data-breach/introduction-to-feature-detection-and-matching-65e27179885d
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        # Use the Brute-Force matcher to obtain matches
        bf_matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)  # Using Euclidean distance
        matches = bf_matcher.knnMatch(des1, des2, k=2)

        # Define Valid Match: whose distance is less than match_ratio times
        # the distance of the second best nearest neighbor.
        match_ratio = 0.6

        # Pick up valid matches
        valid_matches = []
        for m1, m2 in matches:
            if m1.distance < match_ratio * m2.distance:
                valid_matches.append(m1)

        # At least 4 points are needed to compute Homography
        if len(valid_matches) > 4:
            img1_pts = []
            img2_pts = []
            for match in valid_matches:
                img1_pts.append(kp1[match.queryIdx].pt)
                img2_pts.append(kp2[match.trainIdx].pt)

            # Formalize as matrices (for the sake of computing Homography)
            img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
            img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

            # Compute the Homography matrix
            H, mask = cv2.findHomography(img1_pts, img2_pts,
                                            cv2.RANSAC, 5.0)

            # Recompute the Homography in order to improve robustness
            for i in range(mask.shape[0] - 1, -1, -1):
                if mask[i] == 0:
                    np.delete(img1_pts, [i], axis=0)
                    np.delete(img2_pts, [i], axis=0)

            H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

            # print(f'{self.min_match_num} < {np.count_nonzero(mask)} < {self.max_match_num}')

            # Continue if image is ideal for stitching
            if self.min_match_num < np.count_nonzero(mask) < self.max_match_num:
                return True, H

        return False, np.eye(3) # Default

# ------ Set variables ------ #

    def set_debug_state(self, debug_bool: bool) -> None:
        '''
        Use: debug_bool = True
        '''
        self.debug = debug_bool

    def set_filepath(self, file_path: str) -> None:
        '''
        Set the file path to the panoraminc video.
        '''
        self.__filepath = file_path

    def set_min_match_num(self, min_num: Union[int, float]) -> None:
        '''
        Set the min number of kep point matches for homography.
        '''
        self.min_match_num = min_num

    def set_max_match_num(self, max_num: Union[int, float]) -> None:
        '''
        Set the max number of kep point matches for homography.
        '''
        self.max_match_num = max_num

    def set_f(self, f: int) -> None:
        '''
        Set focal length of camera.
        '''
        self.f = f

    def set_resize_factor(self, resize_factor: int) -> None:
        '''
        Set the resize factor.
        '''
        self.__resize = resize_factor
        self.set_f(int(self.f / self.__resize))
        self.set_max_match_num(int(self.max_match_num / weighted_resize(self.__resize)))
        # self.set_min_match_num(int(self.min_match_num / weighted_resize(self.__resize ** (1/self.__resize))))

    def get_frame_dump(self) -> Tuple[bool, list[Union[np.ndarray, None]]]:
        '''
        Return success boolean and the frame dump or an empty list
        '''
        if len(self.frame_dump) > 0:
            return True, self.frame_dump
        else:
            return False, list()

    def get_resize_factor(self) -> int:
        '''
        Returns the resize factor.
        '''
        return self.__resize

    def get_fps(self) -> float:
        '''
        Returns the video's FPS.
        '''

        return self.FPS

    def set_window(self, window: sg.Window):
        '''
        Set the window after creation.
        '''
        self.window = window

    def reset_stitcher(self):
        '''
        Reset the stitcher.
        '''
        del self.frame_dump
        self.frame_dump = []

        del self.panorama_frames
        self.panorama_frames = []

        del self.__pano
        self.__pano = None

# ------ Adjust videostream ------ #

    def locate_frames(self, panorama, frames):
        """Match frames to locations on panorama"""
        # Tuples of (frames, their corresponding locations)
        frames_locations = []

        debug_count = 1
        num_frames = len(frames)


        self.set_min_match_num(0)
        self.set_max_match_num(np.inf)

        for frame in frames:

            h1, w1 = frame.shape[0:2]
            h2, w2 = panorama.shape[0:2]

            success_H, H = self.find_homography(frame, panorama)

            # Place the frame on top of the panorama and save its edeges' locations
            if success_H:
                
                # print('here%d' % debug_count)
                # Retun value is of (subprocess number, current place, total in subsection, text)
                self.window.write_event_value('-UPDATE PROGRESS BAR-', (3, debug_count, num_frames,
                                                                        f"Placing frame {debug_count}/{num_frames} on top of the panorama"))

                # try:
                frame = cv2.warpPerspective(frame, H, (w2, h2))
                # except cv2.error:
                #     self.window.write_event_value('-UPDATE PROGRESS BAR-', (0,0,1,
                #                         f'resize factor ({self.__resize}) too big or f ({self.f * self.__resize}) too small. Please exit program and try again.'))
                #     return None
            
                original_border = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1, 1, 2) # Top_left, Btm_left, Btm_right, Top_right
                transformed_border = np.int32(cv2.perspectiveTransform(original_border, H))

                debug_count += 1

                frames_locations.append((frame, transformed_border))
            
        return frames_locations


# --- METHODS --- #

def right_stitch(frame_pairs) -> np.ndarray:
    '''
    Gets list of (img1, img2, H), stitches to the right.
    Returns panorama.
    '''
    # print(f"Stitching the right side {len(frame_pairs)}")
    # Overall homography
    H_panorama = np.eye(3)

    panorama = frame_pairs[0][0]
    # panorama = np.flip(panorama, axis=1)
    depth = panorama.shape[2]

    top_y = right_x = -np.inf
    btm_y = left_x = np.inf

    for frame_pair in frame_pairs:
        _, right_img, pair_H = frame_pair
        # right_img = np.flip(right_img, axis=1)

        H_panorama = np.linalg.inv(pair_H) @ H_panorama

        offset, new_size = offset_and_size(panorama, right_img, H_panorama)
        H_panorama = offset @ H_panorama

        rightImg_top_left, rightImg_btm_right = get_corners(right_img, H_panorama) #(x, y)

        right_img = cv2.warpPerspective(right_img, H_panorama, new_size)

        pano_top_left, pano_btm_right = get_corners(panorama, offset) #(x, y)

        panorama = cv2.warpPerspective(panorama, offset, new_size)

        panorama = np.where(
            np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], depth, axis=2) != 0,
            panorama,
            right_img,
        ).astype(np.uint8)

        # right_img[pano_top_left[1]:pano_btm_right[1], pano_top_left[0]:pano_btm_right[0]] = panorama[pano_top_left[1]:pano_btm_right[1], pano_top_left[0]:pano_btm_right[0]]

        top_y = max(rightImg_top_left[1], pano_top_left[1])
        btm_y = min(rightImg_btm_right[1], pano_btm_right[1])
        left_x = min(rightImg_top_left[0], pano_top_left[0])
        right_x = max(rightImg_btm_right[0], pano_btm_right[0])

        # panorama = panorama[top_y:btm_y, left_x:right_x]
    

    # panorama = np.flip(panorama, axis=1)
    return panorama[top_y:btm_y, left_x:right_x]

def left_stitch(frame_pairs) -> np.ndarray:
    '''
    Gets list of (img1, img2, H), stitches to the right.
    Returns panorama.
    '''
    # print(f"Stitching the left side {len(frame_pairs)}")
    # Overall homography
    H_panorama = np.eye(3)
    
    frame_pairs.reverse()

    panorama = frame_pairs[0][1]
    # panorama = np.flip(panorama, axis=1)
    depth = panorama.shape[2]

    top_y = right_x = -np.inf
    btm_y = left_x = np.inf

    for frame_pair in frame_pairs:
        left_img, _, pair_H = frame_pair
        # left_img = np.flip(left_img, axis=1)

        H_panorama = pair_H @ H_panorama

        offset, new_size = offset_and_size(left_img, panorama, H_panorama)
        H_panorama = offset @ H_panorama

        leftImg_top_left, leftImg_btm_right = get_corners(left_img, H_panorama) #(x, y)

        left_img = cv2.warpPerspective(left_img, H_panorama, new_size)

        pano_top_left, pano_btm_right = get_corners(panorama, offset) #(x, y)

        panorama = cv2.warpPerspective(panorama, offset, new_size)

        panorama = np.where(
            np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], depth, axis=2) != 0,
            panorama,
            left_img,
        ).astype(np.uint8)

        # left_img[pano_top_left[1]:pano_btm_right[1], pano_top_left[0]:pano_btm_right[0]] = panorama[pano_top_left[1]:pano_btm_right[1], pano_top_left[0]:pano_btm_right[0]]
        
        top_y = max(leftImg_top_left[1], pano_top_left[1])
        btm_y = min(leftImg_btm_right[1], pano_btm_right[1])
        left_x = min(leftImg_top_left[0], pano_top_left[0])
        right_x = max(leftImg_btm_right[0], pano_btm_right[0])
        
        # panorama = left_img.copy()

    # panorama = np.flip(panorama, axis=1)
    return panorama[top_y:btm_y, left_x:right_x]

def get_corners(img: np.ndarray, homography: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    '''
    Gets a warped image and a corresponding 3x3 homography.
    Returns (top left corner, bottom right corner) of incloused rectangle. Corner(x, y)
    '''

    h1, w1 = img.shape[0:2]

    original_border = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1, 1, 2) # Top_left, Btm_left, Btm_right, Top_right
    transformed_border = np.int32(cv2.perspectiveTransform(original_border, homography))

    top_left, btm_left, btm_right, top_right = transformed_border

    top_y = max([top_left[0][1], top_right[0][1]])
    btm_y = min([btm_left[0][1], btm_right[0][1]])
    left_x = max([top_left[0][0], btm_left[0][0]])
    right_x = min([top_right[0][0], btm_right[0][0]])

    return (left_x, top_y), (right_x, btm_y)


def offset_and_size(img1, img2, H):
    '''
    Returns the added offset: np.nparry and the new size: (width, height).
    '''

    # Get heights and widths of input images
    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]

    # Store the 4 ends of each original canvas
    img1_canvas_orig = np.float32([[0, 0], [0, h1],
                                   [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img2_canvas = np.float32([[0, 0], [0, h2],
                              [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # The 4 ends of (perspective) transformed panorama
    img1_canvas = cv2.perspectiveTransform(img1_canvas_orig, H)

    # Get the coordinate range of output (0.5 is fixed for image completeness)
    output_canvas = np.concatenate((img2_canvas, img1_canvas), axis=0)
    [x_min, y_min] = np.int32(output_canvas.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(output_canvas.max(axis=0).ravel() + 0.5)

    # The output matrix after affine transformation
    offset_array = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]], np.float64)

    # Calculate new size
    size = (x_max - x_min, y_max - y_min)

    return offset_array, size

def weighted_resize(resize_factor: int) -> float:
    '''
    Calculate some weighted factor to devide max/min match num by.
    '''

    return np.exp((resize_factor - 1) / resize_factor)






# def recursive_stitch(panorama: np.ndarray, i: int, frame_pairs: list, stitch_direction: int, H_panorama = np.eye(3)):
#     '''
#     Gets panorama, index i, list of (img1, img2, H), stitch_direction: -1 or 1, homography of panorama H_panorama.
#     Returns panorama and its homography H.
#     '''
#     assert stitch_direction == -1 or stitch_direction == 1, "unexpected value of stitch_direction"

#     # Break condition - outside scope of list
#     if i < 0 or i >= len(frame_pairs):
#         return panorama, H
    
#     # Stitch to the right of panorama
#     if stitch_direction == 1:
#         image = frame_pairs[i][1]
#         pair_H = frame_pairs[i][2]

#         # # Deal with opposite stitching (left to right)
#         # image = np.flip(image, axis=1)
#         # panorama = np.flip(panorama, axis=1)

#         # Apply offset
#         new_H = np.linalg.inv(pair_H) @ H_panorama
#         # new_H = new_H / new_H[2][2]
#         offset, new_size = offset_and_size(image, panorama, new_H)
#         new_H = offset @ new_H

#         image = cv2.warpPerspective(image, new_H, new_size)

#         h1, w1 = panorama.shape[0:2]

#         panorama = cv2.warpPerspective(panorama, offset, new_size)

#         original_border = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1, 1, 2) # Top_left, Btm_left, Btm_right, Top_right
#         transformed_border = np.int32(cv2.perspectiveTransform(original_border, offset))
#         [x_min, y_min] = np.int32(transformed_border.min(axis=0).ravel() - 0.5)
#         [x_max, y_max] = np.int32(transformed_border.max(axis=0).ravel() + 0.5)

#         image[y_min:y_min + h1, x_min:x_min + w1] = panorama[y_min:y_min + h1, x_min:x_min + w1]
#         panorama = image.copy()
#         # panorama = np.where(
#         #     np.logical_and(
#         #         np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
#         #         np.repeat(np.sum(image, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
#         #     ),
#         #     0,
#         #     cv2.addWeighted(0.5, ),
#         # ).astype(np.uint8)
#         # panorama = cv2.addWeighted(panorama, 0.6, image, 0.4, 1.0)

#         popped = frame_pairs.pop(i)
#         # assert popped[1] == image, 'pop works diffrently'
#         num_frames = len(frame_pairs) + 1
#         i = int(num_frames / 2)

#         panorama, H = recursive_stitch(panorama, i, frame_pairs, -1, new_H)
#         return panorama, H

#     if stitch_direction == -1:
#         image = frame_pairs[i][0]
#         pair_H = frame_pairs[i][2]

#         h2, w2 = image.shape[:2]

#         # # Deal with opposite stitching (left to right)
#         # image = np.flip(image, axis=1)
#         # panorama = np.flip(panorama, axis=1)

#         # Apply offset
#         new_H = pair_H @ H_panorama
#         # new_H = new_H / new_H[2][2]
#         offset, new_size = offset_and_size(panorama, image, new_H)
#         new_H = offset @ new_H

#         panorama = cv2.warpPerspective(panorama, new_H, new_size)

#         h1, w1 = image.shape[0:2]
        
#         image = cv2.warpPerspective(image, offset, new_size)

#         original_border = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1, 1, 2) # Top_left, Btm_left, Btm_right, Top_right
#         transformed_border = np.int32(cv2.perspectiveTransform(original_border, offset))
#         [x_min, y_min] = np.int32(transformed_border.min(axis=0).ravel() - 0.5)
#         [x_max, y_max] = np.int32(transformed_border.max(axis=0).ravel() + 0.5)

#         # # Deal with opposite stitching (left to right)
#         # image = np.flip(image, axis=1)
#         # panorama = np.flip(panorama, axis=1)

#         panorama[y_min:y_min + h1, x_min:x_min + w1] = image[y_min:y_min + h1, x_min:x_min + w1]

#         # panorama = image.copy()

#         # panorama = cv2.warpPerspective(panorama, offset, new_size)

#         # panorama = np.where(
#         #     np.logical_and(
#         #         np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
#         #         np.repeat(np.sum(image, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
#         #     ),
#         #     0,
#         #     image + panorama,
#         # ).astype(np.uint8)


#         popped = frame_pairs.pop(i)
#         # assert popped[1] == image, 'pop works diffrently'
#         num_frames = len(frame_pairs) + 1
#         i = int(num_frames / 2)

#         panorama, H = recursive_stitch(panorama, i, frame_pairs, -1, new_H)
#         return panorama, H



# def extract_frames(filepath):
#     """
#     Extract (ideal) frames from the video stream (using cv2)
#     Mainly from this paper:
#     https://github.com/ybhan/Panoramic-Photo-Generation/blob/master/Panoramic-Photo-Generation.pdf
#     """
#     # From https://github.com/ybhan/Panoramic-Photo-Generation
#     # Construct VideoCapture object to get frame-by-frame stream\
#     vid_cap = cv2.VideoCapture(filepath)
  
#     # ORB (Oriented FAST and Rotated BRIEF) a key-point detector
#     # and desciptor to desctibe the overlap between frames
#     orb = cv2.ORB_create()

#     # The first key frame (frame0.jpg) is selected by default
#     success, last = vid_cap.read()
#     assert success, "couldn't read first frame"
#     # last = cylindrical_project(last)
#     # last = cv2.resize(last, (640, 360))
#     if not os.path.isdir('key_frames'):
#         os.mkdir('key_frames')

#     for file_path in glob.glob('key_frames/*.jpg'):
#         try:
#             os.remove(file_path)
#         except OSError as e:
#             print("Error: %s : %s" % (file_path, e.strerror))

#     cv2.imwrite('key_frames/frame0.jpg', last)
#     print("Captured frame0.jpg")
#     count = 1
#     frame_num = 1

#     image = last # An unnecessary assignment

#     stride = 1         # stride for accelerating capturing
#     min_match_num = 30  # minimum number of matches required (to stitch well)
#     max_match_num = 50  # maximum number of matches (to avoid redundant frames)

#     frame_dump = []
#     # frame_dump.append(last)
#     # frame_dump_num = 0
#     # cv2.imwrite('key_frames/frame_dump/frame%d.jpg' % frame_dump_num, last)
#     # frame_dump_num += 1

#     while success:
#         # image = cylindrical_project(image)
#         frame_dump.append(image)
#         # cv2.imwrite('key_frames/frame_dump/frame%d.jpg' % frame_dump_num, image)
#         # frame_dump_num += 1
#         if count % stride == 0:
#             # image = cylindrical_project(image)

#             h1, w1 = last.shape[0:2]
#             h2, w2 = image.shape[0:2]

#             # last_crop = last[:, w1 - w2:]  # Crop the right part of img1 for detecting SIFT
#             # last_crop = crop_black(last_crop)
#             # diff = np.size(last, axis=1) - np.size(last_crop, axis=1)

#             # Detect and compute key points and descriptors:
#             # https://medium.com/data-breach/introduction-to-feature-detection-and-matching-65e27179885d
#             kp1, des1 = orb.detectAndCompute(last, None)
#             kp2, des2 = orb.detectAndCompute(image, None)

#             # Use the Brute-Force matcher to obtain matches
#             bf_matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)  # Using Euclidean distance
#             matches = bf_matcher.knnMatch(des1, des2, k=2)

#             # Define Valid Match: whose distance is less than match_ratio times
#             # the distance of the second best nearest neighbor.
#             match_ratio = 0.6

#             # Pick up valid matches
#             valid_matches = []
#             for m1, m2 in matches:
#                 if m1.distance < match_ratio * m2.distance:
#                     valid_matches.append(m1)

#             # At least 4 points are needed to compute Homography
#             if len(valid_matches) > 4:
#                 img1_pts = []
#                 img2_pts = []
#                 for match in valid_matches:
#                     img1_pts.append(kp1[match.queryIdx].pt)
#                     img2_pts.append(kp2[match.trainIdx].pt)

#                 # Formalize as matrices (for the sake of computing Homography)
#                 img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
#                 # img1_pts[:, :, 0] += diff  # Recover its original coordinates
#                 img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

#                 # Compute the Homography matrix
#                 H, mask = cv2.findHomography(img1_pts, img2_pts,
#                                              cv2.RANSAC, 5.0)

#                 # Recompute the Homography in order to improve robustness
#                 for i in range(mask.shape[0] - 1, -1, -1):
#                     if mask[i] == 0:
#                         np.delete(img1_pts, [i], axis=0)
#                         np.delete(img2_pts, [i], axis=0)

#                 H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
#                 #TODO: Figure out why H is garbage on 4th frame. Found a hack, shorten video
#                 #       hacked again, combine in batches, left side and right side. Didn't work :)

#                 # Continue if image is ideal for stitching
#                 if min_match_num < np.count_nonzero(mask) < max_match_num:
#                     # Stitch last (the panorama) with the chosen ideal image
#                     tic = time.time()
#                     last = stitch_by_H(last, image, H)
#                     toc = time.time()
#                     print(f'Time of stitch_by_H(): {toc-tic}')
#                     # tic = time.time()
#                     # last = crop_black(last)
#                     # toc = time.time()
#                     # print(f'Time of crop_black(): {toc-tic}')

#                     # Save key frame as JPG file
#                     print("Captured frame{}.jpg".format(frame_num))
#                     cv2.imwrite('key_frames/frame%d.jpg' % frame_num, last)
#                     frame_num += 1

#         success, image = vid_cap.read()
#         print(count)
#         count += 1

#     cv2.imwrite("key_frames/panorama.jpg", last)
#     return last, frame_dump


# def crop_black(img):
#     """Crop off the black edges."""
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
#     _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
#                                       cv2.CHAIN_APPROX_SIMPLE)

#     max_area = 0
#     best_rect = (0, 0, 0, 0)

#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         deltaHeight = h - y
#         deltaWidth = w - x

#         area = deltaHeight * deltaWidth

#         if area > max_area and deltaHeight > 0 and deltaWidth > 0:
#             max_area = area
#             best_rect = (x, y, w, h)

#     if max_area > 0:
#         img_crop = img[best_rect[1]:best_rect[1] + best_rect[3],
#                    best_rect[0]:best_rect[0] + best_rect[2]]
#     else:
#         img_crop = img

#     return img_crop


# def stitch_by_H(img1, img2, H):
#     """Use the key points to stitch the images.
#     img1: the image containing frames that have been joint before.
#     img2: the newly selected key frame.
#     H: Homography matrix, usually from compute_homography(img1, img2).
#     """
#     # Get heights and widths of input images
#     h1, w1 = img1.shape[0:2]
#     h2, w2 = img2.shape[0:2]

#     # Store the 4 ends of each original canvas
#     img1_canvas_orig = np.float32([[0, 0], [0, h1],
#                                    [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
#     img2_canvas = np.float32([[0, 0], [0, h2],
#                               [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

#     # The 4 ends of (perspective) transformed img1
#     img1_canvas = cv2.perspectiveTransform(img1_canvas_orig, H)

#     # Get the coordinate range of output (0.5 is fixed for image completeness)
#     output_canvas = np.concatenate((img2_canvas, img1_canvas), axis=0)
#     [x_min, y_min] = np.int32(output_canvas.min(axis=0).ravel() - 0.5)
#     [x_max, y_max] = np.int32(output_canvas.max(axis=0).ravel() + 0.5)

#     # The output matrix after affine transformation
#     transform_array = np.array([[1, 0, -x_min],
#                                 [0, 1, -y_min],
#                                 [0, 0, 1]])

#     # Warp the perspective of img1
#     img_output = cv2.warpPerspective(img1, transform_array.dot(H),
#                                      (x_max - x_min, y_max - y_min))

#     # for i in range(-y_min, h2 - y_min):
#     #     for j in range(-x_min, w2 - x_min):
#     #         if np.any(img2[i + y_min][j + x_min]):
#     #             img_output[i][j] = img2[i + y_min][j + x_min]

#     img_output[-y_min:h2-y_min,-x_min:w2-x_min] = img2
#     img_output = img_output[-y_min:-y_min+y_max]
#     return img_output

    # pointX, pointY are coordinates in planar axis;
    # i, j, k are coordinates in cylindrical axis.
    # for i in range(width):
    #     for j in range(height):
    #         theta = (i - centerX) / f
    #         pointX = int(f * np.tan((i - centerX) / f) + centerX)
    #         pointY = int((j - centerY) / np.cos(theta) + centerY)

    #         for k in range(depth):
    #             if 0 <= pointX < width and 0 <= pointY < height:
    #                 cylindrical_img[j, i, k] = img[pointY, pointX, k]
    #             else:
    #                 cylindrical_img[j, i, k] = 0

    # cylindrical_img = crop_black(cylindrical_img)
    # return cylindrical_img

# def main(input_filepath):
#     """
#     Main hub for all functions
#     """
#     # 2. Extract (ideal) frames from the video stream (using cv2)
#     #    (filepath) => list of images and key points
#     panorama = extract_frames(input_filepath)
#     cv2.imwrite("key_frames/panorama.jpg", panorama)

# #   1. Parse the input video file
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('input_file', nargs='+',
#                         help="path of the video file input")
#     args = parser.parse_args()

#     main(args.input_file[0])
