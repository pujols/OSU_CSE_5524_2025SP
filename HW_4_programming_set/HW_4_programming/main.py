import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


def data_loader(args):
    """
    Output:
        I: the image matrix of size N-by-M-by-3.
            The pixel values are within [0, 1].
            The first dimension is horizontal, left-right (N pixels).
            The second dimension is vertical, bottom-up (M pixels).
            The third dimension is color channels, from Red to Green to Blue.
    """

    if args.data in ["dreese", "lighthouse"]:
        print("Using " + args.data + " photo")
        current_dir = os.getcwd()
        image_path = osp.join(current_dir, 'data', args.data + '.png')
        I = np.asarray(Image.open(image_path)).astype(np.float64)/255
        I = np.transpose(I, (1, 0, 2))
        I = I[:, ::-1, :]

    return I


def load_kernel(args):
    """
    Output:
        kernel: the 2D kernel matrix.
    """

    print("Use " + args.kernel + " kernel")

    if args.kernel == "binomial":
        kernel = np.zeros((9, 9)) # this is a placeholder, and you can remove this line completely
        kernel = np.matmul(np.array([1, 4, 6, 4, 1]).reshape(-1, 1), np.array([1, 4, 6, 4, 1]).reshape(1, -1))
        kernel = kernel / np.sum(kernel) # keep this line to make your filter properly normalized

    return kernel


def Convolution(args, I, kernel):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
    Output:
        I_out: convolved image, a 3D matrix of size N-by-M-by-3
    """

    ## Initiate the output
    if np.size(I.shape) != 3:
        I = I.reshape((I.shape[0], I.shape[1], 1))
    I_out = np.zeros(I.shape)

    ## Zero padding
    R = int((kernel.shape[0]-1)/2)
    I_pad = np.concatenate((np.zeros((R, I.shape[1], 3)), I, np.zeros((R, I.shape[1], 3))), axis=0)
    I_pad = np.concatenate((np.zeros((I_pad.shape[0], R, 3)), I_pad, np.zeros((I_pad.shape[0], R, 3))), axis=1)

    ## Convolution (you may add just one more line without for loops; the fewer foor loops you have; the faster the code runs)
    for c in range(I_pad.shape[2]):
        for n in range(I_out.shape[0]):
            for m in range(I_out.shape[1]):
                I_out[n, m, c] = np.sum(I_pad[n : (n + kernel.shape[0]), m : (m + kernel.shape[1]), c] * kernel[::-1, ::-1])

    return I_out


def Downsampling(args, I, kernel):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
    Output:
        I_out: downsampled image, a 3D matrix of size N/2-by-M/2-by-3
    """

    #### Your job 1 starts here: downsampling ####


    #### Your job 1 ends here: downsampling ####

    return I_out


def Upsampling(args, I, kernel):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
    Output:
        I_out: downsampled image, a 3D matrix of size 2N-by-2M-by-3
    """

    ## Initiate the output
    I_out = np.zeros((int(I.shape[0] * 2), int(I.shape[1] * 2), int(I.shape[2])))

    #### Your job 3 starts here: upsampling ####


    #### Your job 3 ends here: upsampling ####

    return I_out


def Gaussian_pyramid(args, I, kernel, scale):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
        scale: number of times to downsample or upsample
    Output:
        Output: a dictionary, where Output[n] records the image after the n-th downsampling (n starts from 0 and ends at scale)
                Output[0] = I, the original image
    """

    ## Initiate the output
    Output = {}
    Output[0] = I

    print("Perfrom downsampling")
    for n in range(scale):
        #### Your job 2 starts here: create the n-th downsampled image ####


        #### Your job 2 starts here: create the n-th downsampled image ####

    ## Display the sequence of downsampled images
    fig, ax = plt.subplots(nrows = 1, ncols = scale + 1)
    for n in range(scale + 1):
        background = np.zeros(I.shape)
        background[0: Output[n].shape[0], 0: Output[n].shape[1], :] = Output[n]
        ax[n].imshow(np.transpose(background, (1, 0, 2)), origin='lower', interpolation='none', aspect='equal')
        ax[n].set_title('Downsampled_' + str(n))
        if args.save:
            current_dir = os.getcwd()
            np.savez(osp.join(current_dir, 'result', 'Downsampled_' + args.data + '_scale_' + str(n) + '.npz'), m=Output[n])
    if args.save:
        current_dir = os.getcwd()
        plt.savefig(osp.join(current_dir, 'result', 'Downsampled_' + args.data + '.png'))
    if args.display:
        plt.show()
    plt.close(fig)

    return Output


def Laplacian_pyramid(args, G_pyramid, kernel, scale):
    """
    Input:
        scale: number of times to downsample or upsample
        G_pyramid: a dictionary of `scale + 1` images. G_pyramid[0] is the original image, a 3D matrix of size N-by-M-by-3
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
    Output:
        Output: a dictionary, where Output[n] records the residual image between the n-th and (n+1)-th downsampled images
                (n starts from 0 and ends at scale - 1)
    """

    ## Initiate the output
    Output = {}

    print("Computing the residual")
    for n in range(scale):
        #### Your job 4 starts here: create the n-th residual image ####


        #### Your job 4 starts here: create the n-th residual image ####

    ## Display the sequence of residual images
    fig, ax = plt.subplots(nrows = 1, ncols = scale)
    for n in range(scale):
        background = np.zeros(G_pyramid[0].shape)
        background[0: Output[n].shape[0], 0: Output[n].shape[1], :] = Output[n] + 0.5
        ax[n].imshow(np.transpose(background, (1, 0, 2)), origin='lower', interpolation='none', aspect='equal')
        ax[n].set_title('Residual_' + str(n))
        if args.save:
            current_dir = os.getcwd()
            np.savez(osp.join(current_dir, 'result', 'Residual_' + args.data + '_scale_' + str(n) + '.npz'), m=Output[n])
    if args.save:
        current_dir = os.getcwd()
        plt.savefig(osp.join(current_dir, 'result', 'Residual_' + args.data + '.png'))
    if args.display:
        plt.show()
    plt.close(fig)

    return Output


def Image_reconstruction(args, I_small, L_pyramid, kernel, scale):
    """
    Input:
        scale: number of times to downsample or upsample
        I_small: the image after downsampling `scale` times, a 3D matrix of size N/(2^scale)-by-M/(2^scale)-by-3
        L_pyramid: a dictionary of `scale` residual images. L_pyramid[n] records the residual image between the n-th and (n+1)-th downsampled images
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
    Output:
        Output: the reconstructed image, a 3D matrix of size N-by-M-by-3
    """

    print("Perfrom upsampling")
    Output = I_small
    for n in range(scale):
        #### Your job 5 starts here: reconstruct the original image ####


        #### Your job 5 starts here: reconstruct the original image ####

    return Output


def main(args):

    # Number of times to downsample or upsample
    scale = int(args.scale)

    ## Create Gaussian Pyramid
    if int(args.current_step) >= 1:
        print("Load image")
        I = data_loader(args)
        kernel = load_kernel(args)
        G_pyramid = Gaussian_pyramid(args, I, kernel, scale)

    ## Create Laplacian Pyramid
    if int(args.current_step) >= 2:
        L_pyramid = Laplacian_pyramid(args, G_pyramid, kernel, scale)

    ## Reconstruct the original image
    if int(args.current_step) >= 3:
        I_reconstruct = Image_reconstruction(args, G_pyramid[scale], L_pyramid, kernel, scale)

        ## Sanity check
        if np.sum((I - I_reconstruct) ** 2) > (10 ** (-12)): # Check if your reconstruction is good enough
            print("Wrong! You can do better!")
        else:
            print("Your result looks good!")

        ## Display the original and reconstructed images
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(np.transpose(I, (1, 0, 2)), origin='lower', interpolation='none', aspect='equal')
        ax[0].set_title('Original')
        ax[1].imshow(np.transpose(I_reconstruct, (1, 0, 2)), origin='lower', interpolation='none', aspect='equal')
        ax[1].set_title('Reconstructed')
        if args.save:
            current_dir = os.getcwd()
            plt.savefig(osp.join(current_dir, 'result', 'Reconstruct_' + args.data + '.png'))
            np.savez(osp.join(current_dir, 'result', 'Reconstruct_' + args.data + '.npz'), m = I_reconstruct)
        if args.display:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convolution_and_Fourier_and_Filter")
    parser.add_argument('--path', default="data", type=str)
    parser.add_argument('--data', default="none", type=str)
    parser.add_argument('--kernel', default="binomial", type=str)
    parser.add_argument('--scale', default=3, type=int)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--current_step', default=1, type=int)
    args = parser.parse_args()
    main(args)

    # Fill in the other students you collaborate with:
    # e.g., Wei-Lun Chao, chao.209
    #
    #