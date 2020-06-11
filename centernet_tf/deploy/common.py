"""
common function for prediction
"""
import numpy as np
import cv2
import tensorflow as tf

def get_affine_transform(center,
                         scale,
                         output_size,
                         rot=0.,
                         inv=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list) and not isinstance(scale, tuple):
        scale = np.array([scale, scale], dtype=np.float32)

    if not isinstance(output_size, np.ndarray) and not isinstance(output_size, list) and not isinstance(output_size,
                                                                                                        tuple):
        output_size = np.array([output_size, output_size], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    src_h = scale_tmp[1]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_h * -0.5], rot_rad)
    dst_dir = np.array([0, dst_h * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def preprocess_image(image, c, s, tgt_w, tgt_h):
    trans_input = get_affine_transform(c, s, (tgt_w, tgt_h))
    image = cv2.warpAffine(image, trans_input, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    #image = image / 255.0
    #image = np.expand_dims(image, axis = 0)

    #image[..., 0] -= 103.939
    #image[..., 1] -= 116.779
    #image[..., 2] -= 123.68

    return image

def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat

def topk(hm, max_objects=100):
    hm = nms(hm)
    # (b, h * w * c)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    # hm2 = tf.transpose(hm, (0, 3, 1, 2))
    # hm2 = tf.reshape(hm2, (b, c, -1))
    hm = tf.reshape(hm, (b, -1))
    # (b, k), (b, k)
    scores, indices = tf.nn.top_k(hm, k=max_objects)
    # scores2, indices2 = tf.nn.top_k(hm2, k=max_objects)
    # scores2 = tf.reshape(scores2, (b, -1))
    # topk = tf.nn.top_k(scores2, k=max_objects)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys

def decode(hm, wh, reg, max_objects=100, nms=False, flip_test=False, num_classes=1, score_threshold=0.1):
    if flip_test:
        hm = (hm[0:1] + hm[1:2, :, ::-1]) / 2
        wh = (wh[0:1] + wh[1:2, :, ::-1]) / 2
        reg = reg[0:1]
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    # (b, h * w, 2)
    wh = tf.reshape(wh, (b, -1, tf.shape(wh)[-1]))
    # (b, k, 2)
    topk_reg = tf.gather(reg, indices, batch_dims=1)
    # (b, k, 2)
    topk_wh = tf.cast(tf.gather(wh, indices, batch_dims=1), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    return detections
