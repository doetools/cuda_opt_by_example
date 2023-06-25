
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include "../../cuda_opt_basics/data_structure.h"
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include "sobel_kernel.cu"

template <typename T> class ImageIO {
public:
  int width;
  int height;
  int channel;
  T *image;

  ImageIO(char *file_name = "original.png") { open(file_name); }

  ~ImageIO() { stbi_image_free(image); }

  int save(char *save_name, vector<T> image) {
    stbi_write_png(save_name, width, height, channel, image.data(),
                   width * channel);

    return 0;
  }

  int open(char *file_name = "original.png") {
    image = stbi_load(file_name, &width, &height, &channel, 3);

    return 0;
  }
};

template <typename T> class Image : public DataBuffer<T> {

public:
  const int width;
  const int height;
  const int channel;

  vector<T> image = vector<T>(this->size, 0);

  Image(int width, int height, int channel, T *image)
      : DataBuffer<T>(width * height * channel, 1, false, false), width(width),
        height(height), channel(channel) {
    convert_image_data(image);
  }

  int convert_image_data(T *image) {
    int id = 0;
    for (int k = 0; k < channel; k++) {
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          this->c_data[id] = image[i * width * channel + j * channel + k];
          id += 1;
        }
      }
    }

    return 0;
  }

  int create_image_data() {
    int id = 0;
    for (int k = 0; k < channel; k++) {
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          image[i * width * channel + j * channel + k] = this->c_data[id];
          id += 1;
        }
      }
    }

    return 0;
  }
};

int main() {

  ImageIO<uint8_t> image_io = ImageIO<uint8_t>();

  Image<uint8_t> image = Image<uint8_t>(image_io.width, image_io.height,
                                        image_io.channel, image_io.image);
  //   create device memory and copy data
  image.create_device_buffer();
  image.copy_to_device();

  // create grid and blocks
  const size_t WARP_SIZE{32};
  size_t M, N, K;

  N = image.width;
  M = image.height;
  K = image.channel;
  dim3 const blocks{WARP_SIZE, WARP_SIZE, 1};
  dim3 const grids{
      ceiling_div<size_t>(N, WARP_SIZE),
      ceiling_div<size_t>(M, WARP_SIZE),
      ceiling_div<size_t>(K, WARP_SIZE),
  };

  // run kernels
  convolute<uint8_t><<<grids, blocks>>>(image.d_data);

  // save figure
  //   image.create_image_data();
  //   image_io.save("new.png", image.image);

  return 0;
}
