/*
C extension to convert and rgb array into a array of chars containing the
hexidecimal string equivalent.
*/

#include <iostream>
#include <stdint.h>
#include <stdio.h>

extern "C" void rgbtohex(const uint8_t* rgb, size_t n_pix, char* hex)
{
    for (size_t i = 0; i < n_pix; ++i) {
        uint8_t red = rgb[i*4 + 0];
        uint8_t green = rgb[i*4 + 1];
        uint8_t blue = rgb[i*4 + 2];
        snprintf(hex+i*8, 8, "#%02x%02x%02x", red, green, blue);
    }
}