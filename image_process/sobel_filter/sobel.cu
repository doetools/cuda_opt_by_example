#include "../../cuda_opt_basics/data_structure.h"
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

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

  int save(char *file_name = "image.png") {
    create_image_data();

    stbi_write_png(file_name, width, height, channel, image.data(),
                   width * channel);

    return 0;
  }
};

int main() {
  int width, height, bpp;

  uint8_t *rgb_image = stbi_load("original.png", &width, &height, &bpp, 3);

  Image<uint8_t> image(width, height, bpp, rgb_image);

  stbi_image_free(rgb_image);

  image.save();

  return 0;
}
