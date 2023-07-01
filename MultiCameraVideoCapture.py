#!/usr/bin/env python

import copy
import cv2
import threading
import queue
import numpy

from typing import Optional
from vimba import *

from datetime import datetime

FRAME_QUEUE_SIZE = 10000
FRAME_HEIGHT = 2064#640#Actual Max 2064
FRAME_WIDTH = 2464#744 # Actual Max 2464

saved_frames = []

def print_preamble():
    print('////////////////////////////////////////////\n')
    print(flush=True)


def add_camera_id(frame: Frame, cam_id: str) -> Frame:
    # Helper function inserting 'cam_id' into given frame. This function
    # manipulates the original image buffer inside frame object.
    cv2.putText(frame.as_opencv_image(), 'Cam: {}'.format(cam_id), org=(0, 30), fontScale=1,
                color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return frame

def resize_if_required(frame: Frame) -> numpy.ndarray:
    # Helper function resizing the given frame, if it has not the required dimensions.
    # On resizing, the image data is copied and resized, the image inside the frame object
    # is untouched.
    cv_frame = frame.as_opencv_image()

    if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
        cv_frame = cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        cv_frame = cv_frame[..., numpy.newaxis]

    return cv_frame



def create_dummy_frame() -> numpy.ndarray:
    cv_frame = numpy.zeros((50, 640, 1), numpy.uint8)
    cv_frame[:] = 0

    cv2.putText(cv_frame, 'No Stream available. Please connect a Camera.', org=(30, 30),
                fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

    return cv_frame


def try_put_frame(q: queue.Queue, cam: Camera, frame: Optional[Frame]):
    try:
        q.put_nowait((cam.get_id(), frame))

    except queue.Full:
        pass


def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
    # Helper function that tries to set a given value. If setting of the initial value failed
    # it calculates the nearest valid value and sets the result. This function is intended to
    # be used with Height and Width Features because not all Cameras allow the same values
    # for height and width.
    feat = cam.get_feature_by_name(feat_name)

    try:
        feat.set(feat_value)

    except VimbaFeatureError:
        min_, max_ = feat.get_range()
        inc = feat.get_increment()

        if feat_value <= min_:
            val = min_

        elif feat_value >= max_:
            val = max_

        else:
            val = (((feat_value - min_) // inc) * inc) + min_

        feat.set(val)

        msg = ('Camera {}: Failed to set value of Feature \'{}\' to \'{}\': '
               'Using nearest valid value \'{}\'. Note that, this causes resizing '
               'during processing, reducing the frame rate.')
        Log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))


# Thread Objects
class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)

        self.log = Log.get_instance()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, frame: Frame):
        # This method is executed within VimbaC context. All incoming frames
        # are reused for later frame acquisition. If a frame shall be queued, the
        # frame must be copied and the copy must be sent, otherwise the acquired
        # frame will be overridden as soon as the frame is reused.
        if frame.get_status() == FrameStatus.Complete:

            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                try_put_frame(self.frame_queue, cam, frame_cpy)

        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        set_nearest_value(self.cam, 'Height', FRAME_HEIGHT)
        set_nearest_value(self.cam, 'Width', FRAME_WIDTH)

        # Try to enable automatic exposure time setting
        try:
            self.cam.ExposureAuto.set('Once')

        except (AttributeError, VimbaFeatureError):
            self.log.info('Camera {}: Failed to set Feature \'ExposureAuto\'.'.format(
                          self.cam.get_id()))

        self.cam.set_pixel_format(PixelFormat.Bgr8)

    def run(self):
        self.log.info('Thread \'FrameProducer({})\' started.'.format(self.cam.get_id()))

        try:
            with self.cam:
                self.setup_camera()

                try:
                    self.cam.start_streaming(self)
                    self.killswitch.wait()

                finally:
                    self.cam.stop_streaming()

        except VimbaCameraError:
            pass

        finally:
            try_put_frame(self.frame_queue, self.cam, None)

        self.log.info('Thread \'FrameProducer({})\' terminated.'.format(self.cam.get_id()))


class FrameConsumer(threading.Thread):
    def __init__(self, frame_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self.out = {}
        self.frame_counter = {}

    def run(self):
        IMAGE_CAPTION = 'Varkauden Puu video capture: Press <Enter> to exit'
        KEY_CODE_ENTER = 13
        KEY_CODE_SPACE = 32
        frames = {}
        alive = True
        recording = False

        self.log.info('Thread \'FrameConsumer\' started.')

        while alive:
            key = cv2.waitKey(1)

            # Toggle recording state if space key is pressed
            if key == KEY_CODE_SPACE:
                recording = not recording
                if recording:
                    print("Started recording")
                    now = datetime.now()
                    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
                    for cam_id in sorted(frames.keys()):
                        self.out[cam_id] = cv2.VideoWriter(f'output_{cam_id}_{timestamp}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (FRAME_WIDTH, FRAME_HEIGHT))
                        self.frame_counter[cam_id] = 0

            # Update current state by dequeuing all currently available frames.
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break

                if frame:
                    frames[cam_id] = frame
                else:
                    frames.pop(cam_id, None)

                frames_left -= 1

            # Record the frames
            if recording:
                for cam_id in sorted(frames.keys()):
                    if cam_id in self.out:
                        self.out[cam_id].write(frames[cam_id].as_opencv_image())
                        self.frame_counter[cam_id] += 1

                        if self.frame_counter[cam_id] >= 900:
                            self.frame_counter[cam_id] = 0
                            self.out[cam_id].release()
                            print(f'Video from camera {cam_id} saved.')
                            now = datetime.now()
                            self.out[cam_id] = cv2.VideoWriter(f'output_{cam_id}_{now.strftime("%Y-%m-%d_%H-%M-%S")}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (FRAME_WIDTH, FRAME_HEIGHT))

            # Construct image by stitching frames together.
            cv2.namedWindow(IMAGE_CAPTION, cv2.WINDOW_NORMAL)
            if frames:
                cv_images = [resize_if_required(frames[cam_id]) for cam_id in sorted(frames.keys())]
                final_img = numpy.concatenate(cv_images, axis=1)
                cv2.imshow(IMAGE_CAPTION, final_img)

                screen_width = 900
                aspect_ratio = final_img.shape[1] / final_img.shape[0]
                resized_image = cv2.resize(final_img, (screen_width, int(screen_width / aspect_ratio)))
                cv2.imshow(IMAGE_CAPTION, resized_image)
            else:
                cv2.imshow(IMAGE_CAPTION, numpy.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), numpy.uint8))

            if key == KEY_CODE_ENTER:
                alive = False

        # Clean up after ourselves
        for cam_id in self.out:
            self.out[cam_id].release()

        cv2.destroyAllWindows()

        self.log.info('Thread \'FrameConsumer\' stopped.')

class MainThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()

    def __call__(self, cam: Camera, event: CameraEvent):
        # New camera was detected. Create FrameProducer, add it to active FrameProducers
        if event == CameraEvent.Detected:
            with self.producers_lock:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()].start()

        # An existing camera was disconnected, stop associated FrameProducer.
        elif event == CameraEvent.Missing:
            with self.producers_lock:
                producer = self.producers.pop(cam.get_id())
                producer.stop()
                producer.join()

    def run(self):
        log = Log.get_instance()
        consumer = FrameConsumer(self.frame_queue)

        vimba = Vimba.get_instance()
        vimba.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)

        log.info('Thread \'MainThread\' started.')

        with vimba:
            # Construct FrameProducer threads for all detected cameras
            for cam in vimba.get_all_cameras():
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)

            # Start FrameProducer threads
            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()

            # Start and wait for consumer to terminate
            vimba.register_camera_change_handler(self)
            consumer.start()
            consumer.join()
            vimba.unregister_camera_change_handler(self)

            # Stop all FrameProducer threads
            with self.producers_lock:
                # Initiate concurrent shutdown
                for producer in self.producers.values():
                    producer.stop()

                # Wait for shutdown to complete
                for producer in self.producers.values():
                    producer.join()

        log.info('Thread \'MainThread\' terminated.')


if __name__ == '__main__':
    print_preamble()
    main = MainThread()
    main.start()
    main.join()


