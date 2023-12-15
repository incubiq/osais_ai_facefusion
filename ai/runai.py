
##
##      FACEFUSION AI
##

import os
import sys
import argparse
from datetime import datetime
import urllib.request 
from PIL import Image 

sys.path.insert(0, './ai')

## FaceFusion specifics
import platform
import shutil
# import onnxruntime
import tensorflow
import facefusion.choices
import facefusion.globals
from facefusion import metadata, wording
from facefusion.predictor import predict_image, predict_video
from facefusion.processors.frame.core import get_frame_processors_modules, load_frame_processor_module
from facefusion.utilities import is_image, is_video, detect_fps, compress_image, merge_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clear_temp, list_module_names, encode_execution_providers, decode_execution_providers, normalize_output_path

## for calling back OSAIS from AI
gNotifyParams=None
gNotifyCallback=None
    
def limit_resources() -> None:
	# prevent tensorflow memory leak
	gpus = tensorflow.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tensorflow.config.experimental.set_virtual_device_configuration(gpu,
		[
			tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit = 512)
		])
	# limit memory usage
	if facefusion.globals.max_memory:
		memory = facefusion.globals.max_memory * 1024 ** 3
		if platform.system().lower() == 'darwin':
			memory = facefusion.globals.max_memory * 1024 ** 6
		if platform.system().lower() == 'windows':
			import ctypes
			kernel32 = ctypes.windll.kernel32 # type: ignore[attr-defined]
			kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
		else:
			import resource
			resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def conditional_process() -> None:
    try: 
        for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
            if not frame_processor_module.pre_process('output'):
                return False
			
    except Exception as err:
        return False
        
    if is_image(facefusion.globals.target_path):
        process_image()
    if is_video(facefusion.globals.target_path):
        process_video()
    return True

def process_image() -> None:
	# safe content image?
	# if predict_image(facefusion.globals.target_path):
	#	return
	shutil.copy2(facefusion.globals.target_path, facefusion.globals.output_path)
	# process frame
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		print('processing ' + frame_processor_module.NAME)
		frame_processor_module.process_image(facefusion.globals.source_path, facefusion.globals.output_path, facefusion.globals.output_path)
		frame_processor_module.post_process()
	# compress image
	print('compressing_image')
	if not compress_image(facefusion.globals.output_path):
		print('compressing_image_failed')
	# validate image
	if is_image(facefusion.globals.target_path):
		print('processing_image_succeed')
	else:
		print('processing_image_failed')

def watermark_video(target_path : str) -> bool:
    from osais_utils import AddWatermark
    from facefusion.utilities import get_temp_directory_path

    temp_output_video_path = get_temp_directory_path(target_path)
    for filename in os.listdir(temp_output_video_path):
        image1 = Image.open(os.path.join(temp_output_video_path, filename))
        imgRet=AddWatermark(image1, facefusion.globals.watermark)
        imgRet.save(os.path.join(temp_output_video_path, filename),"JPEG")

def process_video() -> None:
	# safe content video?
	# if predict_video(facefusion.globals.target_path):
	# 	return

	fps = detect_fps(facefusion.globals.target_path) if facefusion.globals.keep_fps else 25.0
	# create temp
	print('creating_temp')
	create_temp(facefusion.globals.target_path)
	# extract frames
	if gNotifyCallback:
		gNotifyCallback(gNotifyParams, "Extracting video frames...", 0.18)
	print('extracting_frames_fps')
	extract_frames(facefusion.globals.target_path, fps)

	# process frame
	temp_frame_paths = get_temp_frame_paths(facefusion.globals.target_path)
	if temp_frame_paths:
		for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
			print('processing '+ frame_processor_module.NAME)
			frame_processor_module.process_video(facefusion.globals.source_path, temp_frame_paths)
			frame_processor_module.post_process()
	else:
		print('temp_frames_not_found')
		return

    # watermarking
	if facefusion.globals.watermark:
		if gNotifyCallback:
			gNotifyCallback(gNotifyParams, "Watermarking video...", 0.85)
		print('watermarking video...')
		watermark_video(facefusion.globals.target_path)
	
    # merge video
	if gNotifyCallback:
		gNotifyCallback(gNotifyParams, "Finalizing...", 0.95)
	print('merging_video_fps')
	if not merge_video(facefusion.globals.target_path, fps):
		print('merging_video_failed')
		return
	# handle audio
	if facefusion.globals.skip_audio:
		print('skipping_audio')
		move_temp(facefusion.globals.target_path, facefusion.globals.output_path)
	else:
		print('restoring_audio')
		if not restore_audio(facefusion.globals.target_path, facefusion.globals.output_path):
			print('restoring_audio_failed')
			move_temp(facefusion.globals.target_path, facefusion.globals.output_path)

	# clear temp
	print('clearing_temp')
	clear_temp(facefusion.globals.target_path)

	# validate video
	if is_video(facefusion.globals.target_path):
		print('processing_video_succeed')
	else:
		print('processing_video_failed')


## where to save the user profile?
def fnGetUserdataPath(_username):
    _path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_PROFILE_DIR = os.path.join(_path, '_profile')
    USER_PROFILE_DIR = os.path.join(DEFAULT_PROFILE_DIR, _username)
    return {
        "location": USER_PROFILE_DIR,
        "voice": False,
        "picture": True
    }

## WARMUP Data
def getWarmupData(_id):
    try:
        import time
        from werkzeug.datastructures import MultiDict
        ts=int(time.time())
        sample_args = MultiDict([
            ('-u', 'test_user'),
            ('-uid', str(ts)),
            ('-t', _id),
            ('-cycle', '0'),
            ('-o', 'warmup.mp4'),
            ('-filename', 'warmup.jpg'),
            ('-video', 'warmup.mp4')
        ])
        return sample_args
    except Exception as err:
        print("Could not call warm up!\r\n")
        return None

## Notifications from AI
def setNotifyCallback(cb, _aParams): 
    global gNotifyParams
    global gNotifyCallback

    gNotifyParams=_aParams
    gNotifyCallback=cb

## RUN AI
def fnRun(_args): 
    global gNotifyParams
    global gNotifyCallback

    from werkzeug.datastructures import MultiDict

    try:
        vq_parser = argparse.ArgumentParser()

        # OSAIS arguments
        vq_parser.add_argument("-odir", "--outdir", type=str, help="Output directory", default="./_output/", dest='outdir')
        vq_parser.add_argument("-idir", "--indir", type=str, help="input directory", default="./_input/", dest='indir')
        vq_parser.add_argument("-watermark",    "--watermark", type=str, help="watermark filename", default=None, dest='watermark')

        # Add the FaceFusion arguments
        # general
        vq_parser.add_argument('-filename', '--source', type=str, help = 'source_help',  default="warmup.jpg", dest = 'source_path')
        vq_parser.add_argument('-video', '--target', type=str, help = 'target_help',  default="warmup.mp4", dest = 'target_path')
        vq_parser.add_argument("-o", "--output", type=str, help="Output filename", default="output.mp4", dest='output_path')

        #misc
        vq_parser.add_argument('--skip-download', help = 'skip_download_help', dest = 'skip_download', default = "False" )
        vq_parser.add_argument('--headless', help = 'headless_help', dest = 'headless', default = "True")

        # execution
        vq_parser.add_argument('--execution-providers', help = 'execution_providers_help', dest = 'execution_providers', default = [ 'cuda' ], choices = ["cpu", "cuda", "tensorrt"], nargs = '+')
        vq_parser.add_argument('--execution-thread-count', help = 'execution_thread_count_help', dest = 'execution_thread_count', type = int, default = 1)
        vq_parser.add_argument('--execution-queue-count', help = 'execution_queue_count_help', dest = 'execution_queue_count', type = int, default = 1)
        vq_parser.add_argument('--max-memory', help='max_memory_help', dest='max_memory', type = int)

        # face recognition
        vq_parser.add_argument('--face-recognition', help = 'face_recognition_help', dest = 'face_recognition', default = 'reference', choices = facefusion.choices.face_recognitions)
        vq_parser.add_argument('--face-analyser-direction', help = 'face_analyser_direction_help', dest = 'face_analyser_direction', default = 'left-right', choices = facefusion.choices.face_analyser_directions)
        vq_parser.add_argument('--face-analyser-age', help = 'face_analyser_age_help', dest = 'face_analyser_age', choices = facefusion.choices.face_analyser_ages)
        vq_parser.add_argument('--face-analyser-gender', help = 'face_analyser_gender_help', dest = 'face_analyser_gender', choices = facefusion.choices.face_analyser_genders)
        vq_parser.add_argument('--reference-face-position', help = 'reference_face_position_help', dest = 'reference_face_position', type = int, default = 0)
        vq_parser.add_argument('--reference-face-distance', help = 'reference_face_distance_help', dest = 'reference_face_distance', type = float, default = 1.25)
        vq_parser.add_argument('--reference-frame-number', help = 'reference_frame_number_help', dest = 'reference_frame_number', type = int, default = 0)

        # frame extraction
        vq_parser.add_argument('--trim-frame-start', help = 'trim_frame_start_help', dest = 'trim_frame_start', type = int)
        vq_parser.add_argument('--trim-frame-end', help = 'trim_frame_end_help', dest = 'trim_frame_end', type = int)
        vq_parser.add_argument('--temp-frame-format', help = 'temp_frame_format_help', dest = 'temp_frame_format', default = 'jpg', choices = facefusion.choices.temp_frame_formats)
        vq_parser.add_argument('--temp-frame-quality', help = 'temp_frame_quality_help', dest = 'temp_frame_quality', type = int, default = 100, choices = range(101), metavar = '[0-100]')
        vq_parser.add_argument('--keep-temp', help = 'keep_temp_help', dest = 'keep_temp', default = "False")
        
        # output creation
        vq_parser.add_argument('--output-image-quality', help='output_image_quality_help', dest = 'output_image_quality', type = int, default = 95, choices = range(101), metavar = '[0-100]')
        vq_parser.add_argument('--output-video-encoder', help = 'output_video_encoder_help', dest = 'output_video_encoder', default = 'libx264', choices = facefusion.choices.output_video_encoders)
        vq_parser.add_argument('--output-video-quality', help = 'output_video_quality_help', dest = 'output_video_quality', type = int, default = 95, choices = range(101), metavar = '[0-100]')
        vq_parser.add_argument('--keep-fps', help = 'keep_fps_help', dest = 'keep_fps', default = "True")
        vq_parser.add_argument('--skip-audio', help = 'skip_audio_help', dest = 'skip_audio', default = "False")

        vq_parser.add_argument("-res", "--res", type=int, help="resolution", default=256, dest='resolution')
        vq_parser.add_argument("-cimg", "--batch_size", type=int, help="How many output", default=1, dest='batch_size')

        # frame processors
        vq_parser.add_argument('--frame-processors', help = 'frame_processors_help', dest = 'frame_processors', default = [ 'face_swapper' ], nargs = '+')
        program = argparse.ArgumentParser()

        _path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _PrcPath=os.path.join(_path, 'ai/facefusion/processors/frame/modules')
        available_frame_processors = list_module_names(_PrcPath)
        for frame_processor in available_frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            frame_processor_module.register_args(program)
	    
            ## !!! had top Fkin hack the original code, so if upgrading... this will break
            _dict = MultiDict([])
            _tmpArgs = program.parse_args(_dict)
            frame_processor_module.apply_argsAlt(_tmpArgs)

        args = vq_parser.parse_args(_args)
        print(args)

        # general
        src_path = os.path.join(args.indir, args.source_path) 
        tgt_path = os.path.join(args.indir, args.target_path) 
        out_path = os.path.join(args.outdir, args.output_path) 
        facefusion.globals.source_path = src_path
        facefusion.globals.target_path = tgt_path
        facefusion.globals.watermark=None

        if True: # args.watermark:
            try:
                _fileWatermark=os.path.join(args.indir,"keep_watermark.png")
                # urllib.request.urlretrieve(args.watermark, _fileWatermark)
                imgWatermark = Image.open(_fileWatermark)
                facefusion.globals.watermark = imgWatermark
            except:
                print("Could not access Watermark image")
		
        facefusion.globals.output_path = normalize_output_path(facefusion.globals.source_path, facefusion.globals.target_path, out_path)
        # misc
        facefusion.globals.skip_download = (args.skip_download == "True" or args.skip_download == "true")
        facefusion.globals.headless = (args.headless== "True" or args.headless == "true")
        # execution
        facefusion.globals.execution_providers = decode_execution_providers(args.execution_providers)
        facefusion.globals.execution_thread_count = args.execution_thread_count
        facefusion.globals.execution_queue_count = args.execution_queue_count
        facefusion.globals.max_memory = args.max_memory
        # face recognition
        facefusion.globals.face_recognition = args.face_recognition
        facefusion.globals.face_analyser_direction = args.face_analyser_direction
        facefusion.globals.face_analyser_age = args.face_analyser_age
        facefusion.globals.face_analyser_gender = args.face_analyser_gender
        facefusion.globals.reference_face_position = args.reference_face_position
        facefusion.globals.reference_face_distance = args.reference_face_distance
        facefusion.globals.reference_frame_number = args.reference_frame_number
        # frame extraction
        facefusion.globals.trim_frame_start = args.trim_frame_start
        facefusion.globals.trim_frame_end = args.trim_frame_end
        facefusion.globals.temp_frame_format = args.temp_frame_format
        facefusion.globals.temp_frame_quality = args.temp_frame_quality
        facefusion.globals.keep_temp = (args.keep_temp== "True" or args.keep_temp == "true")
        # output creation
        facefusion.globals.output_image_quality = args.output_image_quality
        facefusion.globals.output_video_encoder = args.output_video_encoder
        facefusion.globals.output_video_quality = args.output_video_quality
        facefusion.globals.keep_fps = (args.keep_fps== "True" or args.keep_fps == "true")
        facefusion.globals.skip_audio = (args.skip_audio== "True" or args.skip_audio == "true")
        # frame processors
        facefusion.globals.frame_processors = args.frame_processors
        # uis
        ## facefusion.globals.ui_layouts = args.ui_layouts

        ## our globals
        facefusion.globals.fnNotifyCallback=gNotifyCallback
        facefusion.globals.objNotifyParams=gNotifyParams

        ##
        beg_date = datetime.utcnow()

        limit_resources()
        for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
            if not frame_processor_module.pre_check():
                return
			

        if not conditional_process():
            raise AttributeError("Could not process")

        sys.stdout.flush()

        ## return output
        end_date = datetime.utcnow()
        return {
            "beg_date": beg_date,
            "end_date": end_date,
            "aFile": [facefusion.globals.output_path]
        }
    
    except Exception as err:
        print("\r\nCRITICAL ERROR!!!")
        raise err

