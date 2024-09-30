#ifndef _IMAGE_H
#define _IMAGE_H

#include <cstdint>
#include <cstddef>
#include <fstream>
#include <jpeglib.h>
#include <jerror.h>
#include <memory>
typedef struct
{
	size_t width, height;
	uint8_t *lpData;
} rawimage;

#endif