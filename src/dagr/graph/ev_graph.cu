#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "spiral.h"
#include <vector>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")


__global__ void fill_edges_cuda_kernel(
  const int32_t* __restrict__ batch,
  const int32_t* __restrict__ pos,
  const int32_t* __restrict__ all_timestamps,
  const int32_t* __restrict__ indices,
  const int32_t* __restrict__ event_queue,
        int64_t* __restrict__ edges,
  //      int64_t* __restrict__ num_neighbors_array,
  int B, int Q, int H, int W, int N, int K, float radius, float delta_t_us, int max_num_neighbors, int min_index
)
{
  // linear index
  const int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // check that thread is not out of valid range
  if (event_idx >= N)
    return;

  int radius_int = radius;
  int num_neighbors = 0;

  int offset = event_idx * max_num_neighbors;

  int b        = batch[event_idx];
  int x        = pos[3 * event_idx + 0];
  int y        = pos[3 * event_idx + 1];
  int ts_event = pos[3 * event_idx + 2];

  // first add self edge
  edges[offset + num_neighbors + K * 0] = indices[event_idx]-min_index;
  edges[offset + num_neighbors + K * 1] = indices[event_idx]-min_index;
  num_neighbors++;

  SpiralOut spiral;
  for (int i=0; i<std::pow(2*radius_int+1, 2); i++) {
    if (num_neighbors >= max_num_neighbors) break;
    for (int q=0; q<Q; q++) {
      int x_neighbor = x + spiral.x;
      int y_neighbor = y + spiral.y;

      // break if out of fov
      if (!((x_neighbor >= 0) && (y_neighbor >= 0) && (x_neighbor < W) && (y_neighbor < H))) break;

      int64_t queue_idx = x_neighbor + W * y_neighbor + H * W * q + H * W * Q * b;
      int idx = event_queue[queue_idx];

      // break if exceeded max num neighbors or no more events in queue
      if (idx < min_index) break;

      if (indices[event_idx] > idx) {
          int32_t ts_neighbor = all_timestamps[idx-min_index];
          int32_t dt_us = ts_event - ts_neighbor;

          // if delta t is too large, no edge is added
          if (dt_us > delta_t_us) continue;

          edges[offset + num_neighbors + K * 0] = idx-min_index;
          edges[offset + num_neighbors + K * 1] = indices[event_idx]-min_index;
          num_neighbors++;
          if (num_neighbors >= max_num_neighbors) break;
      }
    }
    spiral.goNext();
  }
  //num_neighbors_array[event_idx] = num_neighbors;
}

void fill_edges_cuda(
    const torch::Tensor& batch,      // N
    const torch::Tensor& pos,      // N x 3
    const torch::Tensor& all_timestamps, // N
    const torch::Tensor& event_queue, // B x Q x H x W
    const torch::Tensor& indices,     // N
    const int max_num_neighbors,
    const float radius,
    const float delta_t_us,
    torch::Tensor& edges,              // 2 x E
    const int min_index
  )
{
  CHECK_INPUT(batch);
  CHECK_INPUT(pos);
  CHECK_INPUT(event_queue);
  CHECK_INPUT(all_timestamps);
  CHECK_INPUT(edges);
  CHECK_INPUT(indices);

  CHECK_DEVICE(batch, event_queue);
  CHECK_DEVICE(batch, pos);
  CHECK_DEVICE(batch, edges);
  CHECK_DEVICE(batch, indices);
  CHECK_DEVICE(batch, all_timestamps);

  unsigned N = batch.size(0);
  unsigned B = event_queue.size(0);
  unsigned Q = event_queue.size(1);
  unsigned H = event_queue.size(2);
  unsigned W = event_queue.size(3);
  unsigned K = edges.size(1);

  unsigned threads = 256;
  dim3 blocks((N + threads - 1) / threads, 1);

  fill_edges_cuda_kernel<<<blocks, threads>>>(
      batch.data<int32_t>(),
      pos.data<int32_t>(),
      all_timestamps.data<int32_t>(),
      indices.data<int32_t>(),
      event_queue.data<int32_t>(),
      edges.data<int64_t>(),
      //num_neighbors.data<int64_t>(),
      B, Q, H, W, N, K, radius, delta_t_us, max_num_neighbors, min_index
    );
}

template <typename scalar_t>
__global__ void insert_in_queue_single_cuda_kernel(
  const scalar_t* __restrict__ indices,
  const scalar_t* __restrict__ events,
  scalar_t* __restrict__ queue,
  int B, int Q, int H, int W, int K
)
{
  // linear index
  const int lin_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // check that thread is not out of valid range
  if (lin_idx >= K)
    return;

  // find out how many events to write, and what is the offset
  int counts = 1;
  int offset = 0;

  // find out the x, y coords where to write the indices
  int x = events[0];
  int y = events[1];
  int b = 0;

  // write indices. break if queue size or counter is exceeded
  for (int q=Q-1; q>=0; q--) {
    int index = b * H * W * Q + q * H * W + y * W + x;
    // for the current position, get the one at q - shift.
    // if q - shift goes in the negative, take from indices instead
    if (q >= counts) {
      int shifted_index = b * H * W * Q + (q-counts) * H * W + y * W + x;
      queue[index] = queue[shifted_index];
    } else {
      queue[index] = indices[offset + counts - 1 - q];
    } 
  }
}


template <typename scalar_t>
__global__ void insert_in_queue_cuda_kernel(
  const scalar_t* __restrict__ indices,
  const scalar_t* __restrict__ unique_coords,
  const scalar_t* __restrict__ cumsum_counts,
  scalar_t* __restrict__ queue,
  int B, int Q, int H, int W, int K
)
{
  // linear index
  const int lin_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // check that thread is not out of valid range
  if (lin_idx >= K)
    return;

  // find out how many events to write, and what is the offset
  int counts, offset;
  if (lin_idx > 0) {
    offset = cumsum_counts[lin_idx-1];
    counts = cumsum_counts[lin_idx] - offset;
  } else {
    offset = 0;
    counts = cumsum_counts[lin_idx];
  }

  // find out the x, y coords where to write the indices
  int x = unique_coords[lin_idx] % W;
  int y = ((unique_coords[lin_idx] - x)/ W) % H;
  int b = unique_coords[lin_idx] / (W*H);

  // write indices. break if queue size or counter is exceeded
  for (int q=Q-1; q>=0; q--) {
    int index = b * H * W * Q + q * H * W + y * W + x;
    // for the current position, get the one at q - shift.
    // if q - shift goes in the negative, take from indices instead
    if (q >= counts) {
      int shifted_index = b * H * W * Q + (q-counts) * H * W + y * W + x;
      queue[index] = queue[shifted_index];
    } else {
      queue[index] = indices[offset + counts - 1 - q];
    } 
  }
}


torch::Tensor insert_in_queue_single_cuda(
    const torch::Tensor& indices,       // 1
    const torch::Tensor& events, // 4 x 1
    const torch::Tensor& queue          // B x Q x H x W
  )
{
  unsigned W = queue.size(3);
  unsigned H = queue.size(2);
  unsigned Q = queue.size(1);
  unsigned B = queue.size(0);
  unsigned K = 1;

  unsigned threads = 256;
  dim3 blocks((K + threads - 1) / threads, 1);

  insert_in_queue_single_cuda_kernel<int32_t><<<blocks, threads>>>(
      indices.data<int32_t>(),
      events.data<int32_t>(),
      queue.data<int32_t>(),
      B, Q, H, W, K
    );

  return queue;
}


torch::Tensor insert_in_queue_cuda(
    const torch::Tensor& indices,       // N -> num events
    const torch::Tensor& unique_coords, // K -> num active pixels
    const torch::Tensor& cumsum_counts, // K -> num active pixels
    const torch::Tensor& queue          // B x Q x H x W
  )
{
  CHECK_INPUT(indices);
  CHECK_INPUT(unique_coords);
  CHECK_INPUT(cumsum_counts);
  CHECK_INPUT(queue);

  CHECK_DEVICE(indices, queue);
  CHECK_DEVICE(indices, unique_coords);
  CHECK_DEVICE(indices, cumsum_counts);
  CHECK_DEVICE(indices, queue);

  unsigned W = queue.size(3);
  unsigned H = queue.size(2);
  unsigned Q = queue.size(1);
  unsigned B = queue.size(0);
  unsigned K = unique_coords.size(0);

  unsigned threads = 256;
  dim3 blocks((K + threads - 1) / threads, 1);

  insert_in_queue_cuda_kernel<int32_t><<<blocks, threads>>>(
      indices.data<int32_t>(),
      unique_coords.data<int32_t>(),
      cumsum_counts.data<int32_t>(),
      queue.data<int32_t>(),
      B, Q, H, W, K
    );

  return queue;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fill_edges_cuda", &fill_edges_cuda, "Find edges from a queue of events.");
  m.def("insert_in_queue_cuda", &insert_in_queue_cuda, "Insert events into queue.");
  m.def("insert_in_queue_single_cuda", &insert_in_queue_single_cuda, "Insert single events into queue.");
}
