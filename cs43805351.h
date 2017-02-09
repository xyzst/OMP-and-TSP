/*
Header file for CS 4380 / CS 5351

Copyright (c) 2016, Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is not permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#ifndef CS43805351
#define CS43805351

#include <stdio.h>
#include <assert.h>

static void writeBMP(const int x, const int y, const unsigned char* const bmp, const char* const name)
{
  const unsigned char bmphdr[54] = {66, 77, 255, 255, 255, 255, 0, 0, 0, 0, 54, 4, 0, 0, 40, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 1, 0, 8, 0, 0, 0, 0, 0, 255, 255, 255, 255, 196, 14, 0, 0, 196, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  unsigned char hdr[1078];
  int i, j, c, xcorr, diff;
  FILE* f;

  xcorr = (x + 3) >> 2 << 2;  // BMPs have to be a multiple of 4 pixels wide.
  diff = xcorr - x;

  for (i = 0; i < 54; i++) hdr[i] = bmphdr[i];
  *((int*)(&hdr[18])) = xcorr;
  *((int*)(&hdr[22])) = y;
  *((int*)(&hdr[34])) = xcorr * y;
  *((int*)(&hdr[2])) = xcorr * y + 1078;
  for (i = 0; i < 256; i++) {
    j = i * 4 + 54;
    hdr[j+0] = i;  // blue
    hdr[j+1] = i;  // green
    hdr[j+2] = i;  // red
    hdr[j+3] = 0;  // dummy
  }

  f = fopen(name, "wb");  assert(f != NULL);
  c = fwrite(hdr, 1, 1078, f);  assert(c == 1078);
  if (diff == 0) {
    c = fwrite(bmp, 1, x * y, f);  assert(c == x * y);
  } else {
    *((int*)(&hdr[0])) = 0;  // need up to three zero bytes
    for (j = 0; j < y; j++) {
      c = fwrite(&bmp[j * x], 1, x, f);  assert(c == x);
      c = fwrite(hdr, 1, diff, f);  assert(c == diff);
    }
  }
  fclose(f);
}

static void line(const int x1, const int y1, const int x2, const int y2, const unsigned char col, unsigned char* const pic, const int width)
{
  assert((0 <= x1) && (0 <= y1) && (0 <= x2) && (0 <= y2));
  assert((x1 < width) && (y1 < width) && (x2 < width) && (y2 < width));

  const int dx = x2 - x1, dy = y2 - y1, dx2 = 2 * dx, dy2 = 2 * dy;

  if (dx == 0) {
    if (dy < 0) {
      for (int y = y2; y <= y1; y++) pic[y * width + x1] = col;
    } else {
      for (int y = y1; y <= y2; y++) pic[y * width + x1] = col;
    }
  } else if (dy == 0) {
    if (dx < 0) {
      for (int x = x2; x <= x1; x++) pic[y1 * width + x] = col;
    } else {
      for (int x = x1; x <= x2; x++) pic[y1 * width + x] = col;
    }
  } else {
    int x = x1, y = y1, d = 0;
    pic[y * width + x] = col;
    if (dx > 0) {
      if (dy > 0) {
        if (dx > dy) {
          do {
            x++; d += dy2;
            if (d >= dx) {y++; d -= dx2;}
            pic[y * width + x] = col;
          } while (x != x2);
        } else {
          do {
            y++; d += dx2;
            if (d >= dy) {x++; d -= dy2;}
            pic[y * width + x] = col;
          } while (y != y2);
        }
      } else {
        if (dx > -dy) {
          do {
            x++; d -= dy2;
            if (d > dx) {y--; d -= dx2;}
            pic[y * width + x] = col;
          } while (x != x2);
        } else {
          do {
            y--; d -= dx2;
            if (d < dy) {x++; d -= dy2;}
            pic[y * width + x] = col;
          } while (y != y2);
        }
      }
    } else {
      if (dy > 0) {
        if (-dx > dy) {
          do {
            x--; d -= dy2;
            if (d <= dx) {y++; d -= dx2;}
            pic[y * width + x] = col;
          } while (x != x2);
        } else {
          do {
            y++; d -= dx2;
            if (d >= dy) {x--; d -= dy2;}
            pic[y * width + x] = col;
          } while (y != y2);
        }
      } else {
        if (dx <= dy) {
          do {
            x--; d += dy2;
            if (d < dx) {y--; d -= dx2;}
            pic[y * width + x] = col;
          } while (x != x2);
        } else {
          do {
            y--; d += dx2;
            if (d < dy) {x--; d -= dy2;}
            pic[y * width + x] = col;
          } while (y != y2);
        }
      }
    }
  }
}

#endif

