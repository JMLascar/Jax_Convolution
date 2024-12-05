## FFT ##
import jax
import jax.numpy as jnp 
import numpy as np 
from jax.experimental import sparse as jaxsparse


### FFT CONVOLUTION## 
def pad_fft_kernel(ker,data_shape):
    """
    A function that pads, shifts, and transforms a convolution kernel
    in preparation for convolution in the Fourier domain. 
    Destined for 1D, 2D, or 1D-2D convolution. 
    
    INPUT:
    * ker: array or tuple of array (jax or numpy).  
    - if array: Convolution kernel (time domain), 1D or 2D
    - if tuple: (1D convolution kernel, 2D convolution kernel)
    * data_shape: int, or tuple. The shape of the data that will be convolved. (Needed for padding.)
    If cube, the shape order is (spectral channels, rows in pixels, columns in pixels). 
    
    OUTPUT: 
    * kerfft: the kernel (frequency domain) ready for convolution in the Fourier domain. 
    - if ker was a tuple, kerfft=(ker1D_FFT,ker2D_FFT), the 2D and 1D kernels (frequency domain). 
    Jax array, or tuple of jax arrays. 
    """
    if type(ker) is tuple:
        ker1D,ker2D=ker
        ker1D=ker1D/ker1D.sum()
        ker2D=ker2D/ker2D.sum()
        ker_l=ker1D.shape[0]
        ker_m,ker_n=ker2D.shape
        L,M,N=data_shape
        assert ker_l%2==1 and  ker_m%2==1 and ker_n%2==1, "the shape of the kernels must all be odd." 
        #Pad the kernels
        ker1D_padded=jnp.pad(ker1D,((L-1)//2+1,(L-1)-(L-1)//2),mode='constant',constant_values=0)
        ker2D_padded=jnp.pad(ker2D,(((M-1)//2+1,(M-1)-(M-1)//2),
                               ((N-1)//2+1,(N-1)-(N-1)//2)),mode='constant',constant_values=0)
        #Shift the kernels
        ker1D_shifted=jnp.fft.ifftshift(ker1D_padded)
        ker2D_shifted=jnp.fft.ifftshift(ker2D_padded)
        return jnp.fft.fft(ker1D_shifted),jnp.fft.fft2(ker2D_shifted)
    else:
        if len(ker.shape) == 1:
            ker1D=ker/ker.sum()
            ker_l=ker1D.shape[0]
            assert ker_l%2==1, "the shape of the kernel must be odd." 
            L=data_shape[0]
            ker1D_padded=jnp.pad(ker1D,((L-1)//2+1,(L-1)-(L-1)//2),mode='constant',constant_values=0)
            ker1D_shifted=jnp.fft.ifftshift(ker1D_padded)
            return jnp.fft.fft(ker1D_shifted)
        elif len(ker.shape) == 2:
            ker2D=ker/ker.sum()
            ker_m,ker_n=ker2D.shape
            assert ker_m%2==1 and ker_n%2==1, "the shape of the kernel must be odd." 
            M,N=data_shape[-2],data_shape[-1]
            ker2D_padded=jnp.pad(ker2D,(((M-1)//2+1,(M-1)-(M-1)//2),
                               ((N-1)//2+1,(N-1)-(N-1)//2)),mode='constant',constant_values=0)
            ker2D_shifted=jnp.fft.ifftshift(ker2D_padded)
            return jnp.fft.fft2(ker2D_shifted)
    

def pad_data_ker1D(data,data_shape,ker_shape):
    """
    A function that pads a vector/image/cube in preparation for convolution in the Fourier domain. 
    Destined for 1D, 2D, or 1D-2D convolution. 
    
    INPUT:
    * data: Jax array. 1D, 2D, or 3D. Data to be convolved. 
    * data_shape: the shape of data (data.shape). 
    If 3D, the shape order is (spectral channels, rows in pixels, columns in pixels). 
    (NB — This is a separate argument so that it can be set to static when jit compiling.)
    * ker_shape: int, or tuple. The shape of the unpadded kernel(s) the data will be convolved with. 
    
    OUTPUT: 
    * padded data: Jax array. The data (time domain) ready to be transformed in Fourier, 
    then for convolution in the Fourier domain. 
    """
    if len(data_shape)==1:
        ker_l=ker_shape[0]
        L=data_shape[0]
        Lp=L+ker_l#-1
        pad_l=((ker_l-1)//2)
        return jnp.zeros(Lp).at[pad_l+1:-(pad_l)].set(data)
    elif len(data_shape)==3:
        L,M,N=data_shape
        ker_l=ker_shape[0]
        Lp=L+ker_l#-1
        pad_l=((ker_l-1)//2)
        return jnp.zeros((Lp,M,N)).at[pad_l+1:-(pad_l),:,:].set(data)

def pad_data_ker2D(data,data_shape,ker_shape):
    """
    A function that pads a vector/image/cube in preparation for convolution in the Fourier domain. 
    Destined for 1D, 2D, or 1D-2D convolution. 
    
    INPUT:
    * data: Jax array. 1D, 2D, or 3D. Data to be convolved. 
    * data_shape: the shape of data (data.shape). 
    If 3D, the shape order is (spectral channels, rows in pixels, columns in pixels). 
    (NB — This is a separate argument so that it can be set to static when jit compiling.)
    * ker_shape: int, or tuple. The shape of the unpadded kernel(s) the data will be convolved with. 
    
    OUTPUT: 
    * padded data: Jax array. The data (time domain) ready to be transformed in Fourier, 
    then for convolution in the Fourier domain. 
    """
    if len(data_shape)==1:
        ker_l=ker_shape[0]
        L=data_shape[0]
        Lp=L+ker_l#-1
        pad_l=((ker_l-1)//2)
        return jnp.zeros(Lp).at[pad_l+1:-(pad_l)].set(data)
    elif len(data_shape)==2:
        ker_m,ker_n=ker_shape
        M,N=data_shape
        Mp,Np=M+ker_m,N+ker_n
        pad_m,pad_n=(ker_m-1)//2,(ker_n-1)//2
        return jnp.zeros((Mp,Np)).at[pad_m+1:-(pad_m),pad_n+1:-(pad_n)].set(data)
    elif len(data_shape)==3:
        L,M,N=data_shape
        if len(ker_shape)==1:
            ker_l=ker_shape[0]
            Lp=L+ker_l#-1
            pad_l=((ker_l-1)//2)
            return jnp.zeros((Lp,M,N)).at[pad_l+1:-(pad_l),:,:].set(data)
        elif len(ker_shape)==2:
            ker_m,ker_n=ker_shape
            Mp,Np=M+ker_m,N+ker_n
            pad_m,pad_n=(ker_m-1)//2,(ker_n-1)//2
            return jnp.zeros((L,Mp,Np)).at[:,pad_m+1:-(pad_m),pad_n+1:-(pad_n)].set(data)
    else:
        raise ValueError("'data' argument must be a 1D, 2D, or 3D array.")

def convolve_from_ker1D_fft_unjit(data,ker_fft,data_shape,ker_shape):
    """
    INPUT:
    * data: Jax array. 1D, 2D, or 3D. Unpadded data to be convolved.
    * ker_fft: the padded, FFT kernel, as obtained by pad_fft_kernel(). 
    * data_shape: the shape of the unpadded data (data.shape). 
    If 3D, the shape order is (spectral channels, rows in pixels, columns in pixels). 
    * ker_shape: int, or tuple. The shape of the unpadded kernel(s) the data will be convolved with. 
    If 1D-2D, (ker_1D.shape, ker_2D.shape). 
    (NB — Shapes are taken as arguments so that they can be set to static when jit compiling.)
    
    """
    padded_data=pad_data_ker1D(data,data_shape,ker_shape)
    if len(data_shape)==1:
        pad_l=(ker_shape[0]-1)//2
        return jnp.fft.ifft(ker_fft*jnp.fft.fft(padded_data))[pad_l+1:-(pad_l)]
    elif len(data_shape)==3:
        pad_l=((ker_shape[0]-1)//2)
        return (jnp.fft.ifft(ker_fft[:,jnp.newaxis,jnp.newaxis]*jnp.fft.fft(padded_data,axis=0),axis=0)
                   [pad_l+1:-(pad_l),:,:])
    else:
        raise ValueError("'data' argument must be a 1D or 3D array.")
convolve_from_ker1D_fft=jax.jit(convolve_from_ker1D_fft_unjit,static_argnames=["data_shape","ker_shape"])

def convolve_from_ker2D_fft_unjit(data,ker_fft,data_shape,ker_shape):
    """
    INPUT:
    * data: Jax array. 1D, 2D, or 3D. Unpadded data to be convolved.
    * ker_fft: the padded, FFT kernel, as obtained by pad_fft_kernel(). 
    * data_shape: the shape of the unpadded data (data.shape). 
    If 3D, the shape order is (spectral channels, rows in pixels, columns in pixels). 
    * ker_shape: int, or tuple. The shape of the unpadded kernel(s) the data will be convolved with. 
    If 1D-2D, (ker_1D.shape, ker_2D.shape). 
    (NB — Shapes are taken as arguments so that they can be set to static when jit compiling.)
    
    """
    padded_data=pad_data_ker2D(data,data_shape,ker_shape)
    if len(data_shape)==2:
        pad_m,pad_n=(ker_shape[0]-1)//2,(ker_shape[1]-1)//2
        return jnp.fft.ifft2(ker_fft*jnp.fft.fft2(padded_data))[pad_m+1:-(pad_m),pad_n+1:-(pad_n)]

    elif len(data_shape)==3:
        pad_m,pad_n=(ker_shape[0]-1)//2,(ker_shape[1]-1)//2
        return (jnp.fft.ifft2(ker_fft[jnp.newaxis,:,:]*jnp.fft.fft2(padded_data))
                   [:,pad_m+1:-(pad_m),pad_n+1:-(pad_n)])
    else:
        raise ValueError("'data' argument must be a 2D or 3D array.")
convolve_from_ker2D_fft=jax.jit(convolve_from_ker2D_fft_unjit,static_argnames=["data_shape","ker_shape"])
#Argh. I just realized — if I want this to be iterable and not be re-compiled every time, I need different functions... 
#So, I *can* make the data-shape static. But not the ker_shape. Need different functions.

def convolve_fft(data,ker):
    """
    All-in-one convolution function using the FFT. 
    If convolution with the same kernel are repeated at many iterations, 
    it is more efficient to only calculate the FFT of the kernel once. 
    Use instead convolve_from_ker_fft. 
    
    INPUT
    * data: Data to be convolved. Jax array. Vector, Image, or Cube. 
    If cube, the shape order is (spectral channels, rows in pixels, columns in pixels). 
    * ker: kernel to convolve the data with. Jax array. Vector or Image. 
    
    OUTPUT
    * convolved array. 
    """
    data_shape=data.shape
    ker_shape=ker.shape
    ker_fft=pad_fft_kernel(ker,data_shape)
    if len(ker_shape)==1:
        return convolve_from_ker1D_fft(data,ker_fft,data_shape,ker_shape)
    elif len(ker_shape)==2:
        return convolve_from_ker2D_fft(data,ker_fft,data_shape,ker_shape)



#### 2D MATRIX CONVOLUTION ####
def generate_2D_conv_matrix(ker,data_shape,sparse=False):
    """
    A function to generate a matrix of convolution, from a 2D kernel, or an array of 2D kernels. 
    INPUT:
    * ker: Array, either 2-dimensional (stationary convolution) 
    or 4-dimensional (non-stationary convolution). The convolution kernel. 
    If non-stationary, the dimensions are (ker_m x ker_n x M x N), 
    where (ker_m,ker_n) are the kernel height and width, 
    and (M,N) are the image height and width.
    * data_shape: Tuple. The shape of the (unpadded) data to be convolved. 
    If cube, the order is (spectral channel x pixel x pixel). 
    * Sparse: Boolean. If True, the matrix will be in sparse BCOO form. 

    OUTPUT: 
    * Array, or sparse array. The matrix of convolution. 
    """
    ker_m,ker_n=ker.shape[0],ker.shape[1]
    M,N=data_shape[-2],data_shape[-1]
    padded_shape_M,padded_shape_N=M+ker_m-1,N+ker_n-1
    MAT_byhand=np.zeros((M,N,padded_shape_M,padded_shape_M))
    for i in (range(M)):
        for j in range(N):
            if ker.ndim==4:
                padded_ker=np.pad(ker[:,:,i,j],((i,padded_shape_M-ker_m-i),(j,padded_shape_N-ker_n-j)))
            else:
                padded_ker=np.pad(ker,((i,padded_shape_M-ker_m-i),(j,padded_shape_N-ker_n-j)))
            MAT_byhand[i,j,:,:]=padded_ker
    
    MAT_byhand_vec=np.reshape(MAT_byhand[:,:,ker_m//2:-ker_m//2+1,ker_n//2:-ker_n//2+1],(M*N,M*N))
    if sparse:
        return jaxsparse.BCOO.fromdense(MAT_byhand_vec)
    else:
        return jnp.asarray(MAT_byhand_vec)
        
def input_data_2D_mat(DATA,data_shape,ker_shape):
    #input: Data (jax) L,M,N 
    #returns: k/2+L+k/2,M,N
    ker_m,ker_n=ker_shape
    if len(data_shape)==2:
        M,N=data_shape
        return jnp.reshape(DATA,((M)*N))
    elif len(data_shape)==3:
        L,M,N=data_shape
        return jnp.reshape(DATA,(L,(M)*(N))).T

def output_data_2D_mat(DATA,data_shape):
    if len(data_shape)==2:
        return jnp.reshape(DATA,(data_shape))
    elif len(data_shape)==3:
        return jnp.reshape(DATA.T,(data_shape))

def conv_2D_Mat_unjit(DATA,MAT,data_shape,ker_shape):
    """
    INPUT
    DATA: jax array, an image or data cube of shape channels x pixels x pixels. 
    MAT: jax array or sparse array. Convolution matrix. 
    Should be obtained with generate_2D_matrix.
    data_shape: the shape of the (unpadded) data. 
    ker_shape: the shape of the (unpadded) convolution kernel. 
    
    OUTPUT
    Convolved result. Data cube of data_shape. 
    """
    padded_data=input_data_2D_mat(DATA,data_shape,ker_shape)
    return output_data_2D_mat(MAT@padded_data,data_shape) 
conv_2D_Mat=jax.jit(conv_2D_Mat_unjit,static_argnames=["data_shape","ker_shape"])

#### 1D MATRIX CONVOLUTION ####
def generate_1D_conv_matrix(ker,data_shape,sparse=False):
    """
    A function to generate a matrix of convolution, from a 1D kernel, or an array of 1D kernels. 
    INPUT:
    * ker: Array, either 1-dimensional (stationary convolution) 
    or 2-dimensional (non-stationary convolution). The convolution kernel. 
    If non-stationary, the dimensions are (ker_l x L), 
    where (ker_l) is the kernel size, and L the data size. 
    * data_shape: Tuple. The shape of the (unpadded) data to be convolved. 
    If cube, the order is (spectral channel x pixel x pixel). 
    * Sparse: Boolean. If True, the matrix will be in sparse BCOO form. 

    OUTPUT: 
    * Array, or sparse array. The matrix of convolution. 
    """
    ker_l=ker.shape[0]
    L=data_shape[0]
    MAT_byhand=np.zeros((L,ker_l+L-1))
    if ker.ndim==2:
        Padded_ker=np.pad(ker,((L-1,0),(0,0)))
    else:
        Padded_ker=np.pad(ker,((L-1,0)))
    for i in range(ker_l+L-1):
        for j in range(i):
            if j<L:
                if ker.ndim==2:
                    MAT_byhand[j,(i-1)]=Padded_ker[L+ker_l-1-i+j,j]
                else:
                    MAT_byhand[j,(i-1)]=Padded_ker[L+ker_l-1-i+j]
    MAT_byhand=MAT_byhand[:,ker_l//2:-ker_l//2+1]
    if sparse:
        return jaxsparse.BCOO.fromdense(MAT_byhand)
    else:
        return jnp.asarray(MAT_byhand)

# def pad_data_1D_mat(DATA,data_shape,ker_shape):
#     ker_l=ker_shape[0]
#     if len(data_shape)==1:
#         return DATA
#     elif len(data_shape)==3:
#         L,M,N=data_shape
#         return (jnp.reshape(DATA,(L,M*N)))

# def unpad_data_1D_mat(DATA,data_shape):
#     if len(data_shape)==1:
#         return DATA
#     elif len(data_shape)==3:
#         L,M,N=data_shape
#         return jnp.reshape(DATA,(L,M,N))

def conv_1D_Mat_unjit(DATA,MAT,data_shape,ker_shape,sparse=False):
    """
    INPUT
    DATA: jax array, an image or data cube of shape channels x pixels x pixels. 
    MAT: jax array or sparse array. Convolution matrix. 
    Should be obtained with generate_2D_matrix.
    data_shape: the shape of the (unpadded) data. 
    ker_shape: the shape of the (unpadded) convolution kernel. 
    
    OUTPUT
    Convolved result. Data cube of data_shape. 
    """
    #return jnp.tensordot(MAT@pad_data_1D_mat(DATA,data_shape,ker_shape),data_shape) 
    if sparse:
        if len(data_shape)==3:
            return jnp.reshape(MAT@jnp.reshape(DATA,(data_shape[0],data_shape[1]*data_shape[2])),
                data_shape)
        else:
            return MAT@DATA
    else:
        return jnp.tensordot(MAT,DATA,axes=1)
conv_1D_Mat=jax.jit(conv_1D_Mat_unjit,static_argnames=["data_shape","ker_shape","sparse"])

    