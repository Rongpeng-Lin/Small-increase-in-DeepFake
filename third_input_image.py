import numpy as np
import cv2,os,sys,argparse

def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T

def random_warp(pa, coverage, scale = 5, zoom = 1):
    image = cv2.resize(cv2.imread(pa),(256,256))
    assert image.shape == (256, 256, 3)
    range_ = np.linspace(128 - coverage//2, 128 + coverage//2, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T
    mapx = mapx + np.random.normal(size=(5,5), scale=scale)
    mapy = mapy + np.random.normal(size=(5,5), scale=scale)
    interp_mapx = cv2.resize(mapx, (80*zoom,80*zoom))[8*zoom:72*zoom,8*zoom:72*zoom].astype('float32')
    interp_mapy = cv2.resize(mapy, (80*zoom,80*zoom))[8*zoom:72*zoom,8*zoom:72*zoom].astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    src_points = np.stack([mapx.ravel(), mapy.ravel() ], axis=-1)
    dst_points = np.mgrid[0:65*zoom:16*zoom,0:65*zoom:16*zoom].T.reshape(-1,2)
    mat = umeyama(src_points, dst_points, True)[0:2]
    target_image = cv2.warpAffine(image, mat, (64*zoom,64*zoom))
    return target_image,warped_image

def mingle(Ar1,Ar2):
    ar1 = cv2.resize(Ar1,(64,64))
    ar2 = cv2.resize(Ar2,(64,64))
    zeros = np.zeros([64,128,3])
    zeros[:,0:64,:] = ar1
    zeros[:,64:128,:] = ar2
    return zeros


def get_input(args):
    pa = args.pa
    for cls in os.listdir(pa):
        if cls.split('.')[-1]=='txt':
            continue
        for im_name in os.listdir(pa+'/'+cls):
            im_dir = pa+'/'+cls+'/'+im_name
            Target_image,Warped_image = random_warp(im_dir, coverage = 220, scale = 5, zoom = 1)
            mingle_input = mingle(Target_image,Warped_image)
            os.remove(im_dir)
            cv2.imwrite(im_dir,mingle_input)
    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pa', type=str, help='Directory with unaligned images.', default="./data/facenet/src/facesalign")
    return parser.parse_args(argv)

if __name__ == '__main__':
    get_input(parse_arguments(sys.argv[1:]))
