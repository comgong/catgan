import numpy as np
import timeit
from . import poisson_blending
import copy
from scipy import linalg
from six.moves import range
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy import ndimage
from sklearn import preprocessing

from skimage.util import random_noise, pad, crop
from skimage.filters import gaussian
from skimage.transform import rotate, rescale, swirl, warp, AffineTransform
from skimage.exposure import adjust_gamma
from skimage import img_as_ubyte, img_as_float
from skimage.color import rgb2hsv, hsv2rgb
from skimage.morphology import dilation, erosion
from skimage.draw import circle, set_color
import random as random

from PIL import Image, ImageEnhance

if Image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(Image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = Image.HAMMING
    if hasattr(Image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = Image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(Image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = Image.LANCZOS

def normalize_img(x):
    x = 2 * (x - np.amin(x))/(np.amax(x)-np.amin(x)) - 1
    maxval = np.amax(x)
    minval = np.amin(x)
    maxstr = "max val: "+str(maxval)
    minstr = "min val: "+str(minval)
    assert (maxval <= 1), maxstr
    assert (minval >= -1), minstr
    return x

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x,
                    transform_matrix,
                    channel_axis=2,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndimage.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def TF_mammo_calcification(im, min_radius=3, max_radius=5, target=None, dim=224):
    assert len(im.shape) == 3
    h, w, nc = im.shape
    im_2 = img_as_float(copy.deepcopy(im[:,:,0]))
    if nc == 4:
        mask = img_as_float(copy.deepcopy(im[:,:,3]))
    radius = random.uniform(min_radius, max_radius)
    # generate center
    center_h = random.uniform(0+radius, h-radius)
    center_w = random.uniform(0+radius, w-radius)
    rr, cc = circle(center_h,center_w,radius)
    x_coord = random.randint(4, w-4)
    y_coord = random.randint(4, h-4)
    #print rr
    #print cc
    im_2[rr,cc] = 1
    im = np.zeros((dim, dim, nc))
    #im[:,:,0] = im_2
    #im[:,:,1] = im_2
    #im[:,:,2] = im_2
    if nc == 4:
        im[:,:,3] = mask
        im[:,:,0] = im_2
        im[:,:,1] = im_2
        im[:,:,2] = im_2
    else:
        for i in range(nc):
            im[:,:,i] = im_2
    #print np.max(im[:,:,3]), np.min(im[:,:,3])
    assert(im.shape[2]==nc)
    return im

def TF_horizontal_stretch(x, r=0.8, channel_axis=2, fill_mode='reflect', cval=0., target=None):
    assert len(x.shape) == 3
    h, w, nc = x.shape
    h_stretch_matrix = np.array([[r, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(h_stretch_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    assert(x.shape[2]==nc)
    return x

def TF_horizontal_shear(x, r=0.5, channel_axis=2, fill_mode='reflect', cval=0., target=None):
    assert len(x.shape) == 3
    h, w, nc = x.shape
    h_stretch_matrix = np.array([[1, r, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(h_stretch_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    assert(x.shape[2]==nc)
    return x

def TF_possion_noise(x, var, dim=224, target=None):
    # get the first channel
    orig_shape = x.shape
    img = np.squeeze(x[:,:,0])
    noisy_img = random_noise(img, mode='poisson', clip=True)
    #noisy_img = normalize_img(noisy_img)
    im = np.zeros((orig_shape[0], orig_shape[1], orig_shape[2]))
    im[:,:,0] = noisy_img
    im[:,:,1] = noisy_img
    im[:,:,2] = noisy_img
    if orig_shape[2] == 4:
        im[:,:,3] = img_as_float(x[:,:,3])
    #print np.max(im[:,:,0]), np.min(im[:,:,0])
    assert(im.shape[2]==orig_shape[2])
    return im


def TF_salt_and_pepper_noise(x, amount, dim=224, target=None):
    # get the first channel
    orig_shape = x.shape
    img = np.squeeze(x[:,:,0])
    noisy_img = random_noise(img, mode='s&p', clip=True)
    #noisy_img = normalize_img(noisy_img)
    im = np.zeros((orig_shape[0], orig_shape[1], orig_shape[2]))
    im[:,:,0] = noisy_img
    im[:,:,1] = noisy_img
    im[:,:,2] = noisy_img
    if orig_shape[2] == 4:
        im[:,:,3] = img_as_float(x[:,:,3])
    #print np.max(im[:,:,0]), np.min(im[:,:,0])
    assert(im.shape[2]==orig_shape[2])
    return im

def TF_mammo_random_crop(im, depth):
    assert len(x.shape) == 3
    h, w, nc = x.shape
    pre_crop_size = image_size*3
    assert ((h >= pre_crop_size) and (w >= pre_crop_size))
    im = np.array(np.resize(im,(pre_crop_size, pre_crop_size, nc)))
    start_indices = [0, image_size, 2*image_size]
    x_start = random.choice(start_indices)
    y_start = random.choice(start_indices)
    im = im[y_start:y_start+image_size,x_start:x_start+image_size,:]
    return im

def np_to_pil(x):
    """Converts from np image in skimage float format to PIL.Image"""
    # maxval = np.amax(x)
    # minval = np.amin(x)
    # maxstr = "max val: "+str(maxval)
    # assert (maxval <= 1), maxstr
    # assert (minval >= -1), "min val is smaller than -1"
    x = np.squeeze(np.uint8(img_as_ubyte(x)))
    return Image.fromarray(np.uint8(img_as_ubyte(x)))

def pil_to_np(x):
    """Converts from PIL.Image to np float format image"""
    x = np.asarray(x)
    if len(x.shape) == 2: 
        x = x[:,:,np.newaxis]
    return img_as_float(np.asarray(x))

def TF_mammo_enhance_contrast_3D(x, p=1.0, target=None):
    assert len(x.shape) == 3
    h, w, nc = x.shape
    one_layer = preprocessing.minmax_scale(np.squeeze(x[:,:,0]), feature_range=(-0.99, 0.99))
    maxval = np.amax(one_layer)
    minval = np.amin(one_layer)
    maxstr = "max val: "+str(maxval)
    assert (maxval <= 1), maxstr
    assert (minval >= -1), "min val is smaller than -1"
    enhancer = ImageEnhance.Contrast(np_to_pil(one_layer))
    one_layer = pil_to_np(enhancer.enhance(p))
    for i in range(nc-1):
        x[:,:,i] = np.squeeze(one_layer)
    return x

def TF_mammo_rotate(x, angle=0.0, target=None):
    assert len(x.shape) == 3
    h, w, nc = x.shape

    # Rotate using edge fill mode
    return rotate(x, angle, mode='edge', order=1)

def TF_mammo_rotate_random(x, rg, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0., target=None):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.deg2rad(np.random.uniform(-rg, rg))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def TF_mammo_rotate_discrete(x, theta, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0., target=None):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    orig_shape = x.shape
    #assert(orig_shape[2]==4)
    if orig_shape[2] == 4: #masks
        mask = img_as_float(copy.deepcopy(x[:,:,3]))
    rad_theta = np.deg2rad(theta)
    #print rad_theta
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def TF_mammo_zoom_random(x, zoom_range, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0., target=None):

    if zoom_range == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(1-zoom_range, 1+zoom_range, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def TF_mammo_random_horizontal_flip(x):
    if np.random.random() < 0.5:
        x = flip_axis(x, 1) # 1 is the column axis
    return x

def TF_mammo_horizontal_flip(x, target=None):
    x = flip_axis(x, 1) 
    return x

def TF_mammo_vertical_flip(x, target=None):
    x = flip_axis(x, 0) # 0 is the row axis axis
    return x

def TF_mammo_zoom(x, scale=1.0, target=None):
    assert len(x.shape) == 3
    h, w, nc = x.shape
    assert h == w

    # Zoom
    xc   = rescale(x, scale)
    diff = h - xc.shape[0]
    d    = int(np.floor(diff / 2.0))
    if d >= 0:
        padding = ((d, d),(d, d),(0,0))
        if diff % 2 != 0:
            padding = ((d,d+1), (d,d + 1),(0,0))
        return np.pad(xc, padding, mode='edge')
    else:
        return xc[-d:h-d, -d:w-d].reshape(h, w, nc)

def TF_mammo_brightness(x, N=500, target=None): 
    #compute mean of top N pixels
    assert len(x.shape) == 3
    x_new = np.uint8(img_as_ubyte(x))
    topN = sorted(x_new.flatten())[-N:]
    meanN = np.mean(topN)

    #compute upper bound and brightenss enhancement
    upper_bound = 255 / meanN 
    p = np.random.choice(np.linspace(1.0, upper_bound, num=10))
    enhancer = ImageEnhance.Brightness(np_to_pil(x))
    return pil_to_np(enhancer.enhance(p))

def compute_bb(args, num_pixels, upper_bound=100): 
    h0, h1 = min(args[0]), max(args[0])
    w0, w1 = min(args[1]), max(args[1])
    h, w = h1-h0, w1-w0 
    
    h_lo = max(2,h0-num_pixels)
    h_hi = min(upper_bound-2,h1+num_pixels) 
    w_lo = max(2,w0-num_pixels)
    w_hi = min(upper_bound-2,w1+num_pixels) 
    h = h_hi - h_lo
    w = w_hi - w_lo

    center = (h_lo + h / 2, w_lo + w / 2)
    
    return [h_lo, h_hi, w_lo, w_hi], \
        [h0, h1, w0, w1], center

def rotate_image(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def TF_translate_structure_with_tissue(x, translation=None, num_pixels=10, \
    target=None, dim=224):
    #create copy of target
    print("FROM TRANSLATE TARGET SIZE ", target.shape)
    print("FROM TRANSLATE X SIZE", x.shape)
    start = timeit.default_timer()
    imsg = img_as_float(copy.deepcopy(x))
    target = img_as_float(copy.deepcopy(target))
    print("GOT TO TF TRANSLATE 1")
    
    #translate/rotate/dilate segmentation
    args = np.where(imsg[:,:,1] != 0)
    # inspect values of segmentation
    bb, seg, _ = compute_bb(args, num_pixels, upper_bound=dim)
    #print "BB: ", bb
    #print "SEG: ", seg
    h_lo, h_hi, w_lo, w_hi = bb
    h0, h1, w0, w1 = seg

    if not translation: 
        y_translate = np.random.randint(-h0, dim-h1)
        x_translate = np.random.randint(-w0, dim-w1)
    else:
        y_translate, x_translate = translation

        maxa0, mina0 = max(args[0]), min(args[0])
        maxa1, mina1 = max(args[1]), min(args[1])
        if y_translate + maxa0 >= dim: 
            y_translate = dim - maxa0 - 1
        if x_translate + maxa1 >= dim: 
            x_translate = dim - maxa1 - 1 
        if mina0 + y_translate < 0: 
            y_translate = -mina0
        if mina1 + x_translate < 0: 
            x_translate = -mina1
    
    new_seg_args = [args[0]+y_translate, args[1]+x_translate]
    mass_seg = imsg[h_lo:h_hi, w_lo:w_hi, 0]
    new_mask = np.zeros((dim, dim))
    # HYPOTHESIS DOES THIS NEED TO BE A FLOAT?? YO WTF
    new_mask[h_lo:h_hi,w_lo:w_hi] = 255
    #new_mask = img_as_float(copy.deepcopy(new_mask))
    
    #-39,15
    #print "y_translate: ", y_translate, " x_translate: ", x_translate
    #print "MASK SHAPE BEFORE PASSING: ", new_mask.shape
    print("GOT TO TF TRANSLATE 2 - about to call poisson blending")
    new_img = poisson_blending.blend(target, imsg[:,:,0], \
            new_mask, offset=(y_translate, x_translate))
    print("GOT TO TF TRANSLATE 3 - RETURNED FROM poisson blending")
    new_img = target
    new_seg = np.zeros((dim, dim))
    new_seg[new_seg_args[0],new_seg_args[1]] = 255
    
    im = np.zeros((dim, dim, 2))
    im[:,:,0] = new_img
    im[:,:,1] = img_as_float(new_seg.astype(np.uint8))
    stop = timeit.default_timer()
    print("RETURNING FROM TF TRANSLATE, RUNTIME IN S: ", stop-start)
    return im

def TF_translate_structure_with_tissue_3D(imsg, translation=None, num_pixels=10, \
    target=None, dim=224):
    #create copy of target
    orig_shape = imsg.shape
    imsg = img_as_float(copy.deepcopy(imsg))
    target = img_as_float(copy.deepcopy(target))
    
    #translate/rotate/dilate segmentation
    args = np.where(imsg[:,:,3] != 0)
    bb, seg, _ = compute_bb(args, num_pixels, upper_bound=dim)
    h_lo, h_hi, w_lo, w_hi = bb
    h0, h1, w0, w1 = seg

    if not translation: 
        y_translate = np.random.randint(-h0, dim-h1)
        x_translate = np.random.randint(-w0, dim-w1)
    else:
        y_translate, x_translate = translation

        maxa0, mina0 = max(args[0]), min(args[0])
        maxa1, mina1 = max(args[1]), min(args[1])
        if y_translate + maxa0 >= dim: 
            y_translate = dim - maxa0 - 1
        if x_translate + maxa1 >= dim: 
            x_translate = dim - maxa1 - 1 
        if mina0 + y_translate < 0: 
            y_translate = -mina0
        if mina1 + x_translate < 0: 
            x_translate = -mina1
    
    new_seg_args = [args[0]+y_translate, args[1]+x_translate]
    mass_seg = imsg[h_lo:h_hi, w_lo:w_hi, 0]
    new_mask = np.zeros((dim, dim))
    new_mask[h_lo:h_hi,w_lo:w_hi] = 255
    
    #-39,15
    new_img = poisson_blending.blend(target, imsg[:,:,0], \
            new_mask, offset=(y_translate, x_translate))
    new_img = target
    new_seg = np.zeros((dim, dim))
    new_seg[new_seg_args[0],new_seg_args[1]] = 255
    
    im = np.zeros((dim, dim, 4))
    im[:,:,0] = new_img
    im[:,:,1] = new_img
    im[:,:,2] = new_img
    im[:,:,3] = img_as_float(new_seg.astype(np.uint8))
    assert(im.shape == orig_shape)

    return im

def TF_rotate_structure_with_tissue(imsg, p=None, num_pixels=10, \
    target=None, dim=224):
    #convert format
    imsg = np.uint8(img_as_ubyte(imsg))

    #translate/rotate/dilate segmentation
    args = np.where(imsg[:,:,1] != 0)
    bb, _, center = compute_bb(args, num_pixels, upper_bound=dim)
    h_lo, h_hi, w_lo, w_hi = bb
    
    if not p: 
        p = np.random.randint(360)
        
    mass_seg = imsg[h_lo:h_hi, w_lo:w_hi, 0]
    new_mask = np.zeros((dim, dim))
    new_mask[h_lo:h_hi,w_lo:w_hi] = 255
        
    #rotate 
    center_rev = (center[1], center[0])
    rot_im = rotate_image(imsg[:,:,0], p, center_rev)
    rot_mask = rotate_image(imsg[:,:,1], p, center_rev)    
    
    rot_args = np.where(rot_mask)
    rot_bb, _, _ = compute_bb(rot_args, num_pixels, upper_bound=dim)
    rh_lo, rh_hi, rw_lo, rw_hi = rot_bb
    
    new_mask = np.zeros((dim, dim))
    new_mask[rh_lo:rh_hi, rw_lo:rw_hi] = 255
    ##first: normal, second: full original \
    ##that's been transformed, third: mask \
    ##of the transformed
    new_im = poisson_blending.blend(target, rot_im, new_mask)    
    
    im = np.zeros((dim, dim, 2))
    
    im[:,:,0] = img_as_float(new_im.astype(np.uint8))
    im[:,:,1] = img_as_float(rot_mask.astype(np.uint8))

    return im




