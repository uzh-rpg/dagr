#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")


template <typename scalar_t>
__global__ void masked_isdiff_kernel(
  int64_t* __restrict__ indices,
  const scalar_t* __restrict__ x_old,
  const scalar_t* __restrict__ x_new,
  int K, int C, float atol, float rtol
)
{
  // linear index
  const int lin_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // check that thread is not out of valid range
  if (lin_idx >= K)
    return;

  // find out how many events to write, and what is the offset
  int64_t temp = indices[lin_idx];
  indices[lin_idx] = -1;
  int offset = temp*C;
  for (int i=0; i<C; i++) {
    float input = x_old[offset + i];
    float other = x_new[offset + i];
    if (std::abs(input - other) > atol + rtol * other) {
      indices[lin_idx] = temp;
      break;
    }
  }
}

template <typename scalar_t>
__global__ void masked_inplace_BN_kernel(
  const int64_t* __restrict__ indices,
  const scalar_t* __restrict__ x,
  scalar_t* __restrict__ x_out,
  const scalar_t* __restrict__ running_mean,
  const scalar_t* __restrict__ running_var,
  const scalar_t* __restrict__ weight,
  const scalar_t* __restrict__ bias,
  int K, int C, float eps
)
{
  // linear index
  const int lin_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // check that thread is not out of valid range
  if (lin_idx >= K*C)
    return;

  int i = lin_idx / C;
  int c = lin_idx % C;

  int x_lin_idx = C * indices[i] + c;
  x_out[x_lin_idx] = (x[x_lin_idx] - running_mean[c]) / (sqrt(running_var[c] + eps)) * weight[c] + bias[c];
}

void masked_inplace_BN(
    const torch::Tensor& indices,
    const torch::Tensor& x,
    torch::Tensor& x_out,
    const torch::Tensor& running_mean,
    const torch::Tensor& running_var,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps
  )
{
  unsigned K = indices.size(0);
  unsigned C = x.size(1);

  unsigned threads = 256;
  dim3 blocks((K*C + threads - 1) / threads, 1);

  masked_inplace_BN_kernel<float><<<blocks, threads>>>(
    indices.data<int64_t>(),
    x.data<float>(),
    x_out.data<float>(),
    running_mean.data<float>(),
    running_var.data<float>(),
    weight.data<float>(),
    bias.data<float>(), K, C, eps
    );
}

torch::Tensor masked_isdiff(
    const torch::Tensor& indices, // N -> num events
    const torch::Tensor& x_old,   // K -> num active pixels
    const torch::Tensor& x_new,    // K -> num active pixels
    float atol, float rtol
  )
{
  CHECK_INPUT(indices);
  CHECK_INPUT(x_old);
  CHECK_INPUT(x_new);

  CHECK_DEVICE(indices, x_old);
  CHECK_DEVICE(indices, x_new);

  unsigned K = indices.size(0);
  unsigned C = x_old.size(1);

  unsigned threads = 256;
  dim3 blocks((K + threads - 1) / threads, 1);

  masked_isdiff_kernel<float><<<blocks, threads>>>(
      indices.data<int64_t>(),
      x_old.data<float>(),
      x_new.data<float>(),
      K, C, atol, rtol
    );

  return indices.index({indices > -1});
}


template <typename scalar_t>
__global__ void masked_lin_kernel(
  int64_t* __restrict__ indices,
  const scalar_t* __restrict__ x_in,
  scalar_t* __restrict__ x_out,
  const scalar_t* __restrict__ weight,
  const scalar_t* __restrict__ bias,
  int K, int Cin, int Cout, bool add
)
{
  // linear index
  const int lin_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // check that thread is not out of valid range
  if (lin_idx >= K*Cout)
    return;

  int i = lin_idx / Cout;
  int cout = lin_idx % Cout;

  int x_out_lin_idx = Cout * indices[i] + cout;
  int x_int_lin_idx = Cin * indices[i];

  if (!add)
      x_out[x_out_lin_idx] = 0;

  for (int cin=0; cin<Cin; cin++) {
    x_out[x_out_lin_idx] += x_in[x_int_lin_idx + cin] * weight[cout*Cin + cin];
  }
  x_out[x_out_lin_idx] += bias[cout];
}

template <typename scalar_t>
__global__ void masked_lin_no_bias_kernel(
  int64_t* __restrict__ indices,
  const scalar_t* __restrict__ x_in,
  scalar_t* __restrict__ x_out,
  const scalar_t* __restrict__ weight,
  int K, int Cin, int Cout, bool add
)
{
  // linear index
  const int lin_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // check that thread is not out of valid range
  if (lin_idx >= K*Cout)
    return;

  int i = lin_idx / Cout;
  int cout = lin_idx % Cout;

  int x_out_lin_idx = Cout * indices[i] + cout;
  int x_int_lin_idx = Cin * indices[i];

  if (!add)
      x_out[x_out_lin_idx] = 0;

  for (int cin=0; cin<Cin; cin++) {
    x_out[x_out_lin_idx] += x_in[x_int_lin_idx + cin] * weight[cout*Cin + cin];
  }
}

void masked_lin_no_bias(
    const torch::Tensor& indices,
    const torch::Tensor& x_in,
    torch::Tensor& x_out,
    const torch::Tensor& weight,
    bool add
  )
{
  unsigned K = indices.size(0);
  unsigned Cin = weight.size(1);
  unsigned Cout = weight.size(0);

  unsigned threads = 256;
  dim3 blocks((K*Cout + threads - 1) / threads, 1);

  masked_lin_no_bias_kernel<float><<<blocks, threads>>>(
    indices.data<int64_t>(),
    x_in.data<float>(),
    x_out.data<float>(),
    weight.data<float>(),
    K, Cin, Cout, add);
}


void masked_lin(
    const torch::Tensor& indices,
    const torch::Tensor& x_in,
    torch::Tensor& x_out,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    bool add
  )
{
  unsigned K = indices.size(0);
  unsigned Cin = weight.size(1);
  unsigned Cout = weight.size(0);

  unsigned threads = 256;
  dim3 blocks((K*Cout + threads - 1) / threads, 1);

  masked_lin_kernel<float><<<blocks, threads>>>(
    indices.data<int64_t>(),
    x_in.data<float>(),
    x_out.data<float>(),
    weight.data<float>(),
    bias.data<float>(), K, Cin, Cout, add);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("masked_lin", &masked_lin, "Find edges from a queue of events.");
  m.def("masked_lin_no_bias", &masked_lin_no_bias, "Find edges from a queue of events.");
  m.def("masked_isdiff", &masked_isdiff, "Find edges from a queue of events.");
  m.def("masked_inplace_BN", &masked_inplace_BN, "Find edges from a queue of events.");
}
