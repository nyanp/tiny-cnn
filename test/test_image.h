/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

namespace tiny_dnn {

TEST(image, default_ctor_uint8) {
  image<uint8_t> img;

  EXPECT_TRUE(img.empty());
}

TEST(image, default_ctor_float) {
  image<float> img;

  EXPECT_TRUE(img.empty());
}

TEST(image, copy_ctor) {
  uint8_t a[] = {0, 127, 255};

  image<uint8_t> src(a, 3, 1, image_type::grayscale);
  image<float> dst(src);

  EXPECT_FLOAT_EQ(0.0f, dst.at(0, 0));
  EXPECT_FLOAT_EQ(127.0f, dst.at(1, 0));
  EXPECT_FLOAT_EQ(255.0f, dst.at(2, 0));
}

TEST(image, create_from_array_uint8) {
  uint8_t src[] = {1, 2, 3, 4, 5, 6};

  {
    image<uint8_t> img(src, 3, 2, image_type::grayscale);

    EXPECT_EQ(3u, img.width());
    EXPECT_EQ(2u, img.height());
    EXPECT_EQ(1u, img.depth());
    EXPECT_EQ(int(image_type::grayscale), int(img.type()));
    EXPECT_EQ(5u, img.at(1, 1));
  }

  {
    image<uint8_t> img(src, 1, 2, image_type::rgb);

    EXPECT_EQ(1u, img.width());
    EXPECT_EQ(2u, img.height());
    EXPECT_EQ(3u, img.depth());
    EXPECT_EQ(int(image_type::rgb), int(img.type()));
    EXPECT_EQ(4u, img.at(0, 1, 1));
  }
}

TEST(image, create_from_array_float) {
  float src[] = {1, 2, 3, 4, 5, 6};

  {
    image<float> img(src, 3, 2, image_type::grayscale);

    EXPECT_EQ(3u, img.width());
    EXPECT_EQ(2u, img.height());
    EXPECT_EQ(1u, img.depth());
    EXPECT_EQ(int(image_type::grayscale), int(img.type()));
    EXPECT_FLOAT_EQ(5.0f, img.at(1, 1));
  }

  {
    image<float> img(src, 1, 2, image_type::rgb);

    EXPECT_EQ(1u, img.width());
    EXPECT_EQ(2u, img.height());
    EXPECT_EQ(3u, img.depth());
    EXPECT_EQ(int(image_type::rgb), int(img.type()));
    EXPECT_FLOAT_EQ(4.0f, img.at(0, 1, 1));
  }
}

TEST(image, create_zero_filled_uint8) {
  image<uint8_t> img(shape3d(3, 2, 1), image_type::grayscale);

  EXPECT_EQ(0u, img.at(0, 0));
  EXPECT_EQ(0u, img.at(2, 1));
}

TEST(image, create_zero_filled_float) {
  image<float> img(shape3d(3, 2, 1), image_type::grayscale);

  EXPECT_FLOAT_EQ(0.0f, img.at(0, 0));
  EXPECT_FLOAT_EQ(0.0f, img.at(2, 1));
}

TEST(image, copy) {
  uint8_t base[] = {1, 2, 3, 4, 5, 6};
  image<uint8_t> src(base, 3, 2, image_type::grayscale);
  image<float> dst(src);

  EXPECT_FLOAT_EQ(1.0f, dst.at(0, 0));
  EXPECT_FLOAT_EQ(6.0f, dst.at(2, 1));
}

TEST(image, read_png_1bit) {
  std::string path;

  ASSERT_TRUE(resolve_path("testimage/pngsuite/primary/basi0g01.png", path));

  image<uint8_t> img(path, image_type::grayscale);

  EXPECT_EQ(32u, img.width());
  EXPECT_EQ(32u, img.height());
  EXPECT_EQ(1u, img.depth());
  EXPECT_EQ(255u, img.at(0, 0));
  EXPECT_EQ(255u, img.at(15, 15));
  EXPECT_EQ(0u, img.at(31, 0));
  EXPECT_EQ(0u, img.at(16, 15));
  EXPECT_EQ(0u, img.at(31, 31));
}

TEST(image, read_png_2bit) {
  std::string path;

  ASSERT_TRUE(resolve_path("testimage/pngsuite/primary/basi0g02.png", path));

  image<uint8_t> img(path, image_type::grayscale);

  EXPECT_EQ(32u, img.width());
  EXPECT_EQ(32u, img.height());
  EXPECT_EQ(1u, img.depth());
  EXPECT_EQ(0u, img.at(0, 0));
  EXPECT_EQ(85u, img.at(4, 0));
  EXPECT_EQ(170u, img.at(8, 0));
  EXPECT_EQ(255u, img.at(12, 0));
}

TEST(image, read_png_4bit) {
  std::string path;

  ASSERT_TRUE(resolve_path("testimage/pngsuite/primary/basn0g04.png", path));

  image<uint8_t> img(path, image_type::grayscale);

  EXPECT_EQ(32u, img.width());
  EXPECT_EQ(32u, img.height());
  EXPECT_EQ(1u, img.depth());
  EXPECT_EQ(0u, img.at(0, 0));
  EXPECT_EQ(17u, img.at(4, 0));
  EXPECT_EQ(34u, img.at(8, 0));
  EXPECT_EQ(51u, img.at(12, 0));
}

TEST(image, read_png_8bit) {
  std::string path;

  ASSERT_TRUE(resolve_path("testimage/pngsuite/primary/basn0g08.png", path));

  image<uint8_t> img(path, image_type::grayscale);

  EXPECT_EQ(32u, img.width());
  EXPECT_EQ(32u, img.height());
  EXPECT_EQ(1u, img.depth());
  EXPECT_EQ(0u, img.at(0, 0));
  EXPECT_EQ(32u, img.at(0, 1));
  EXPECT_EQ(64u, img.at(0, 2));
  EXPECT_EQ(96u, img.at(0, 3));
}

TEST(image, read_png_8bit_gray2bgr) {
  std::string path;

  ASSERT_TRUE(resolve_path("testimage/pngsuite/primary/basn0g08.png", path));

  image<uint8_t> img(path, image_type::bgr);

  EXPECT_EQ(32u, img.width());
  EXPECT_EQ(32u, img.height());
  EXPECT_EQ(3u, img.depth());
  EXPECT_EQ(0u, img.at(0, 0));
  EXPECT_EQ(0u, img.at(0, 0, 0));
  EXPECT_EQ(0u, img.at(0, 0, 1));
  EXPECT_EQ(0u, img.at(0, 0, 2));
  EXPECT_EQ(32u, img.at(0, 1));
  EXPECT_EQ(32u, img.at(0, 1, 0));
  EXPECT_EQ(32u, img.at(0, 1, 1));
  EXPECT_EQ(32u, img.at(0, 1, 2));
  EXPECT_EQ(64u, img.at(0, 2));
  EXPECT_EQ(64u, img.at(0, 2, 0));
  EXPECT_EQ(64u, img.at(0, 2, 1));
  EXPECT_EQ(64u, img.at(0, 2, 2));
  EXPECT_EQ(96u, img.at(0, 3));
  EXPECT_EQ(96u, img.at(0, 3, 0));
  EXPECT_EQ(96u, img.at(0, 3, 1));
  EXPECT_EQ(96u, img.at(0, 3, 2));
}

TEST(image, read_png_8bit_rgba) {
  std::string path;

  ASSERT_TRUE(resolve_path("testimage/pngsuite/primary/basn6a08.png", path));
  {
    image<uint8_t> img(path, image_type::bgr);

    EXPECT_EQ(3u, img.depth());  // alpha channel is just ignored

    // bgr order
    EXPECT_EQ(255u, img.at(31, 31, 0));
    EXPECT_EQ(32u, img.at(31, 31, 1));
    EXPECT_EQ(0u, img.at(31, 31, 2));
  }
  {
    image<uint8_t> img(path, image_type::rgb);

    // rgb order
    EXPECT_EQ(0u, img.at(31, 31, 0));
    EXPECT_EQ(32u, img.at(31, 31, 1));
    EXPECT_EQ(255u, img.at(31, 31, 2));
  }
}

TEST(image, read_png_8bit_rgba2gray) {
  std::string path;

  ASSERT_TRUE(resolve_path("testimage/pngsuite/primary/basn6a08.png", path));

  image<uint8_t> img(path, image_type::grayscale);
  image<uint8_t> rgb(path, image_type::rgb);

  EXPECT_EQ(32u, img.width());
  EXPECT_EQ(32u, img.height());
  EXPECT_EQ(1u, img.depth());

  for (size_t y = 0; y < 32; y++) {
    for (size_t x = 0; x < 32; x++) {
      uint8_t r = rgb.at(x, y, 0);
      uint8_t g = rgb.at(x, y, 1);
      uint8_t b = rgb.at(x, y, 2);

      // RGB2GRAY conversion in tiny-dnn (inherited from stbi__compute_y
      // function in stb_image.h)
      // note that weights slightly differ from opencv/matlab
      //
      //  tiny-dnn: 0.300r + 0.586g + 0.113b
      //  opencv:   0.299r + 0.587g + 0.114b
      uint8_t expected_y = (((r * 77) + (g * 150) + (b * 29)) >> 8);

      EXPECT_EQ(expected_y, img.at(x, y));
    }
  }
}

TEST(image, read_bmp_8bit) {
  std::string path;

  ASSERT_TRUE(resolve_path("testimage/bmp/8bit.bmp", path));

  image<uint8_t> img(path, image_type::rgb);

  EXPECT_EQ(255u, img.at(0, 0, 0));
  EXPECT_EQ(255u, img.at(0, 0, 1));
  EXPECT_EQ(255u, img.at(0, 0, 2));

  EXPECT_EQ(0u, img.at(31, 31, 0));
  EXPECT_EQ(0u, img.at(31, 31, 1));
  EXPECT_EQ(255u, img.at(31, 31, 2));
}

TEST(image, read_bmp_8bit_rgb2gray) {
  std::string path;

  ASSERT_TRUE(resolve_path("testimage/bmp/8bit.bmp", path));

  image<uint8_t> img(path, image_type::grayscale);

  uint8_t expected_y = (255 * 29) >> 8;

  EXPECT_EQ(255u, img.at(0, 0));
  EXPECT_EQ(expected_y, img.at(31, 31));
}

TEST(image, read_bmp_24bit) {
  std::string path;

  ASSERT_TRUE(resolve_path("testimage/bmp/24bit.bmp", path));

  {
    image<uint8_t> img(path, image_type::bgr);

    EXPECT_EQ(3u, img.depth());  // alpha channel is just ignored

    // bgr order
    EXPECT_EQ(255u, img.at(31, 31, 0));
    EXPECT_EQ(32u, img.at(31, 31, 1));
    EXPECT_EQ(0u, img.at(31, 31, 2));
  }
  {
    image<uint8_t> img(path, image_type::rgb);

    // rgb order
    EXPECT_EQ(0u, img.at(31, 31, 0));
    EXPECT_EQ(32u, img.at(31, 31, 1));
    EXPECT_EQ(255u, img.at(31, 31, 2));
  }
}

TEST(image, read_bmp_24bit_rgb2gray) {
  std::string path;

  ASSERT_TRUE(resolve_path("testimage/bmp/24bit.bmp", path));

  image<uint8_t> img(path, image_type::grayscale);
  image<uint8_t> rgb(path, image_type::rgb);

  EXPECT_EQ(32u, img.width());
  EXPECT_EQ(32u, img.height());
  EXPECT_EQ(1u, img.depth());

  for (size_t y = 0; y < 32; y++) {
    for (size_t x = 0; x < 32; x++) {
      uint8_t r = rgb.at(x, y, 0);
      uint8_t g = rgb.at(x, y, 1);
      uint8_t b = rgb.at(x, y, 2);

      // RGB2GRAY conversion in tiny-dnn (inherited from stbi__compute_y
      // function in stb_image.h)
      // note that weights slightly differ from opencv/matlab
      //
      //  tiny-dnn: 0.300r + 0.586g + 0.113b
      //  opencv:   0.299r + 0.587g + 0.114b
      uint8_t expected_y = (((r * 77) + (g * 150) + (b * 29)) >> 8);

      EXPECT_EQ(expected_y, img.at(x, y));
    }
  }
}

TEST(image, resize) {
  image<> img(shape3d(10, 10, 3), image_type::rgb);

  img.resize(32, 32);

  EXPECT_EQ(32u, img.width());
  EXPECT_EQ(32u, img.height());
  EXPECT_EQ(3u, img.depth());
  EXPECT_EQ(int(image_type::rgb), int(img.type()));
  EXPECT_EQ(size_t(32 * 32 * 3), img.data().size());
}

TEST(image, empty) {
  image<> img;
  image<> img2(shape3d(1, 1, 1), image_type::grayscale);

  EXPECT_TRUE(img.empty());
  EXPECT_FALSE(img2.empty());
}

TEST(image, fill) {
  image<> img(shape3d(3, 3, 3), image_type::rgb);
  img.fill(127);

  for (auto i : img) {
    EXPECT_EQ(i, 127u);
  }
}

TEST(image, fromrgb) {
  // clang-format off
    std::vector<uint8_t> rgb = {
    //  r    g    b
        0, 127, 255,
        1, 126, 254,
        2, 125, 253,
        3, 124, 252
    };
  // clang-format on

  image<uint8_t> img(shape3d(2, 2, 3), image_type::rgb);

  img.from_rgb(rgb.begin(), rgb.end());

  EXPECT_EQ(0u, img.at(0, 0, 0));
  EXPECT_EQ(127u, img.at(0, 0, 1));
  EXPECT_EQ(255u, img.at(0, 0, 2));

  EXPECT_EQ(1u, img.at(1, 0, 0));
  EXPECT_EQ(126u, img.at(1, 0, 1));
  EXPECT_EQ(254u, img.at(1, 0, 2));

  EXPECT_EQ(2u, img.at(0, 1, 0));
  EXPECT_EQ(125u, img.at(0, 1, 1));
  EXPECT_EQ(253u, img.at(0, 1, 2));

  EXPECT_EQ(3u, img.at(1, 1, 0));
  EXPECT_EQ(124u, img.at(1, 1, 1));
  EXPECT_EQ(252u, img.at(1, 1, 2));
}

TEST(image, torgb) {
  // clang-format off
    std::vector<uint8_t> rgb = {
    //  r    g    b
        0, 127, 255,
        1, 126, 254,
        2, 125, 253,
        3, 124, 252
    };
  // clang-format on

  image<uint8_t> img(shape3d(2, 2, 3), image_type::rgb);

  img.from_rgb(rgb.begin(), rgb.end());

  auto dst = img.to_rgb<uint8_t>();

  for (size_t i = 0; i < rgb.size(); i++) {
    EXPECT_EQ(dst[i], rgb[i]);
  }
}

TEST(image, mean_image) {
  // clang-format off
    std::vector<uint8_t> rgb = {
    //  r    g    b
        0, 127, 255,
        1, 126, 254,
        2, 125, 253,
        3, 124, 252
    };
  // clang-format on

  float_t mean[] = {(0 + 1 + 2 + 3) / 4.0f, (127 + 126 + 125 + 124) / 4.0f,
                    (255 + 254 + 253 + 252) / 4.0f};

  image<uint8_t> img(shape3d(2, 2, 3), image_type::rgb);
  image<float_t> mean_expected(mean, 1, 1, image_type::rgb);

  img.from_rgb(rgb.begin(), rgb.end());
  auto mean_actual = mean_image(img);

  ASSERT_EQ(mean_actual.shape(), shape3d(1, 1, 3));
  EXPECT_FLOAT_EQ(mean_actual[0], mean_expected[0]);
  EXPECT_FLOAT_EQ(mean_actual[1], mean_expected[1]);
  EXPECT_FLOAT_EQ(mean_actual[2], mean_expected[2]);
}

TEST(image, subtract_scalar) {
  uint8_t src_pixels[] = {
    0,   1,   2,   3,   // r
    32,  33,  34,  35,  // g
    252, 253, 254, 255  // b
  };

  uint8_t sub_pixels[] = {1, 15, 254};

  // should be saturated
  uint8_t expected_pixels[] = {0, 0, 1, 2, 17, 18, 19, 20, 0, 0, 0, 1};

  image<uint8_t> src(src_pixels, 2, 2, image_type::rgb);
  image<uint8_t> sub(sub_pixels, 1, 1, image_type::rgb);
  image<uint8_t> expected(expected_pixels, 2, 2, image_type::rgb);

  auto actual = subtract_scalar(src, sub);

  for (size_t i = 0; i < 3; i++) {
    EXPECT_EQ(actual.at(0, 0, i), expected.at(0, 0, i));
    EXPECT_EQ(actual.at(0, 1, i), expected.at(0, 1, i));
    EXPECT_EQ(actual.at(1, 0, i), expected.at(1, 0, i));
    EXPECT_EQ(actual.at(1, 1, i), expected.at(1, 1, i));
  }
}

TEST(image, subtract_image) {
  // clang-format off
    uint8_t src_pixels[] = {
        0, 1, // r
        32,33, // g
        254,255 // b
    };

    uint8_t sub_pixels[] = {
        1, 1,
        31,33,
        0, 250
    };

    // should be saturated
    uint8_t expected_pixels[] = {
        0, 0,
        1, 0,
        254, 5
    };
  // clang-format on

  image<uint8_t> src(src_pixels, 2, 1, image_type::rgb);
  image<uint8_t> sub(sub_pixels, 2, 1, image_type::rgb);
  image<uint8_t> expected(expected_pixels, 2, 1, image_type::rgb);

  auto actual = subtract_image(src, sub);

  for (size_t i = 0; i < 3; i++) {
    EXPECT_EQ(actual.at(0, 0, i), expected.at(0, 0, i));
    EXPECT_EQ(actual.at(1, 0, i), expected.at(1, 0, i));
  }
}

}  // namespace tiny_dnn
