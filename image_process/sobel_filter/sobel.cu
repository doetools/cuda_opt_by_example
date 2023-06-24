#include "../../cuda_opt_basics/data_structure.h"
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

template <typename T> class Image : public DataBuffer<T> {

public:
  const int width;
  const int height;
  const int channel;

  vector<T> image_data = vector<T>(this->size, 0);

  Image(int width, int height, int channel, T *image)
      : DataBuffer<T>(width * height * channel, 1, false, false), width(width),
        height(height), channel(channel) {
    // do something
  }

  int convert_image_data(T *image) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        for (int k = 0; k < CHANNEL_NUM; k++) {
          if (k == 3)
            rgb_image[i * width * CHANNEL_NUM + j * CHANNEL_NUM + k] = 255;

          if (k == 0)
            rgb_image[i * width * CHANNEL_NUM + j * CHANNEL_NUM + k] = 100;

          if (k == 1)
            rgb_image[i * width * CHANNEL_NUM + j * CHANNEL_NUM + k] =
                uint8_t(rand() / 255);
        }
      }
    }

    return 0;
  }

  int create_image_data() { return 0; }
};

int main() {
  int width, height, bpp;

  uint8_t *rgb_image = stbi_load("original.png", &width, &height, &bpp, 3);

  Image<uint8_t> image(width, height, bpp, rgb_image);

  stbi_image_free(rgb_image);

  return 0;
}

int read(char *file_name = "original.png") {}

// int save()
// {
//     int width = 800;
//     int height = 800;
//     int CHANNEL_NUM = 3;

//     vector<uint8_t> rgb_image(width * height * CHANNEL_NUM, 255);

//     int index = 0;

//     for (int i = 0; i < height; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             for (int k = 0; k < CHANNEL_NUM; k++)
//             {
//                 if (k == 3)
//                     rgb_image[i * width * CHANNEL_NUM + j * CHANNEL_NUM + k]
//                     = 255;

//                 if (k == 0)
//                     rgb_image[i * width * CHANNEL_NUM + j * CHANNEL_NUM + k]
//                     = 100;

//                 if (k == 1)
//                     rgb_image[i * width * CHANNEL_NUM + j * CHANNEL_NUM + k]
//                     = uint8_t(rand() / 255);
//             }
//         }
//     }

//     stbi_write_png("image.png", width, height, CHANNEL_NUM, rgb_image.data(),
//     width * CHANNEL_NUM);

//     return 0;
// }