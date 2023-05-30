import os
import cv2
import numpy as np
import onnxruntime
from PIL import Image
import datetime

x = datetime.datetime.now()
model_path = "pretrained/modnet.onnx"

def custom_background(background, foreground):
  x = (background.size[0]-foreground.size[0])/2 + 0.5
  y = (background.size[1]-foreground.size[1])/2 + 0.5
  box = (x, y, foreground.size[0] + x, foreground.size[1] + y)
  crop = background.crop(box)
  final_image = crop.copy()
  paste_box = (0, final_image.size[1] - foreground.size[1], final_image.size[0], final_image.size[1])
  final_image.paste(foreground, paste_box, mask=foreground)
  return final_image

def predict(image_path, savefile, fill_color, mode, bgfilepath):
    if not os.path.exists(image_path):
        print('Cannot find input path: {0}'.format(image_path))
        exit()
    if not os.path.exists(model_path):
        print('Cannot find model path: {0}'.format(model_path))
        exit()

    ref_size = 512

    def get_scale_factor(im_h, im_w, ref_size):

        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        x_scale_factor = im_rw / im_w
        y_scale_factor = im_rh / im_h

        return x_scale_factor, y_scale_factor

    ##############################################
    #  Main Inference part
    ##############################################

    # read image
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # normalize values to scale it between -1 to 1
    im = (im - 127.5) / 127.5

    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w, ref_size)

    # resize image
    im = cv2.resize(im, None, fx = x, fy = y, interpolation = cv2.INTER_AREA)

    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis = 0).astype('float32')

    # Initialize session and get prediction
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    # refine matte
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation = cv2.INTER_AREA)

    im_PIL = Image.open(image_path)
    matte = Image.fromarray(matte)
    im_PIL.putalpha(matte)  
    dir = (os.getcwd()).encode('unicode_escape').decode()

    if mode == 'fillcolor':
        background = Image.new(im_PIL.mode[:-1], im_PIL.size, fill_color)
        background.paste(im_PIL, im_PIL.split()[-1]) 
        im_PIL = background
        im_PIL.save(savefile)
    elif mode == 'transparent':
        im_PIL.save(savefile)
    elif mode == 'image':
        bgim = Image.open(bgfilepath)
        if (bgim.size[0]<im_PIL.size[0] or bgim.size[0]<im_PIL.size[0]):
            bgimRE = bgim.resize((im_PIL.size[0],im_PIL.size[0]))
            im_PIL = custom_background(bgimRE, im_PIL)
        else:
            im_PIL = custom_background(bgim, im_PIL)
        im_PIL.save(savefile)


