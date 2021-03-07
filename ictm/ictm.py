import logging

import numpy as np
import torch as th

from .utils import (convolve, edge_indicator_function,
                    gaussian_kernel_generator, gradient_magnitude_gaussian)

logr = logging.getLogger('ictm')


def chan_vese(im,
              init_seg,
              image_dimension,
              tau=0.02,
              lambd=0.05,
              max_step=100,
              kernel_size=9):
    """Chan vese model solved with iterative convolution-thresholding method
    (ICTM)."""
    def chan_vese_F(im, seg, eps=1e-5):
        C = seg.mul(im).sum().div(seg.sum() + eps)
        return (im - C).pow_(2)

    kern = gaussian_kernel_generator((kernel_size, ) * image_dimension,
                                     (tau, ) * image_dimension).to(im)
    u = init_seg.gt(0).to(im)
    steps = 0
    while steps <= max_step:
        phi = chan_vese_F(im, u) - chan_vese_F(im, 1-u) +  \
                lambd * np.sqrt(np.pi / tau) * \
                convolve((1 - 2*u), kern, image_dimension).squeeze()
        u_next = phi.le(0).to(im)
        if u_next.ne(u).sum() == 0:
            break
        del u
        u = u_next
        steps += 1
    logr.debug(f'Total steps: {steps}. Max steps: {max_step}.')
    return u.to(th.uint8)


def geodesic_active_contour(im,
                            init_seg,
                            image_dimension,
                            sigma=1.,
                            tau=2.,
                            lambd=-0.2,
                            max_step=10,
                            kernel_size=9):
    """Geodesic active contour model solved with iterative convolution-
    thresholding method (ICTM)."""
    # https://arxiv.org/pdf/2007.00525.pdf
    if type(tau) is not list and type(tau) is not tuple:
        tau = (tau, ) * image_dimension
    g = edge_indicator_function(
        im,
        sigma,
        image_dimension,
        gaussian=lambda x, y, z: gradient_magnitude_gaussian(
            x, y, z, kernel_size=kernel_size))
    g_sqrt = g.sqrt()
    kern = gaussian_kernel_generator((kernel_size, ) * image_dimension,
                                     tau).to(g)
    steps = 0
    u = init_seg.gt(0).to(g)
    while steps <= max_step:
        phi = g_sqrt * convolve(g_sqrt *
                                (1 - 2 * u), kern, image_dimension) + lambd * g
        u_next = phi.le(0)
        if u_next.ne(u).sum() == 0:
            break
        del u
        u = u_next
        steps += 1
    logr.debug(f'Total steps: {steps}. Max steps: {max_step}.')
    return u.to(th.uint8)


def local_intensity_fitting(im,
                            init_seg,
                            image_dimension,
                            sigma=1.5,
                            tau=1.,
                            lambd=90,
                            mu1=1.,
                            mu2=1.,
                            eps=0.01,
                            max_step=30,
                            kernel_size=9):
    """Local intensity fitting model solved with iterative convolution-
    thresholding method (ICTM)."""
    kern_sigma = gaussian_kernel_generator((kernel_size, ) * image_dimension,
                                           (sigma, ) * image_dimension).to(im)
    kern_tau = gaussian_kernel_generator((kernel_size, ) * image_dimension,
                                         (tau, ) * image_dimension).to(im)
    u = ((1 - init_seg.gt(0).to(im)) * (1 - eps * 2) + eps)
    steps = 0
    im_sigma = convolve(im, kern_sigma, image_dimension)
    one_sigma = convolve(th.ones_like(im), kern_sigma, image_dimension)
    while steps <= max_step:
        Ik = im * u
        c1 = convolve(u, kern_sigma, image_dimension)
        c2 = convolve(Ik, kern_sigma, image_dimension)
        f1 = c2 / c1
        f2 = (im_sigma - c2) / (one_sigma - c1)
        phi1 = mu1 * convolve(
            f1.pow(2), kern_sigma, image_dimension) - 2 * mu1 * im * convolve(
                f1, kern_sigma, image_dimension) + lambd * convolve(
                    1 - u, kern_tau, image_dimension)
        phi2 = mu2 * convolve(
            f2.pow(2), kern_sigma, image_dimension) - 2 * mu2 * im * convolve(
                f2, kern_sigma, image_dimension) + lambd * convolve(
                    u, kern_tau, image_dimension)
        u_next = phi1.le(phi2).to(im) * (1 - eps * 2) + eps
        if (u_next - u).abs().sum() < eps:
            break
        del u
        u = u_next
        steps += 1
    logr.debug(f'Total steps: {steps}. Max steps: {max_step}.')
    return u.le(eps).to(th.uint8)
