/******************************************************************************
 * The MIT License (MIT)

 * Copyright (c) 2021, NVIDIA CORPORATION.

 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ******************************************************************************/

#pragma once

#include "helper_math.h"
#include <vector_types.h>

using namespace std;

  
#define MAX_LEVELS          15
#define MAX_TOTAL_POINTS    (0x1<<27)

typedef unsigned char       uchar;
typedef unsigned short      ushort;
typedef unsigned int        uint;

typedef unsigned long long  morton_code;
typedef short4              point_data;
typedef long3               tri_index;

typedef struct
{
    float m[4][4];
} float4x4;


static __inline__ __host__ __device__ point_data make_point_data(short x, short y, short z)
{
    point_data p;
    p.x = x; p.y = y; p.z = z;
    return p;
}


static __inline__ __host__ __device__ morton_code ToMorton(point_data V)
{
    morton_code mcode = 0;

    for (uint i = 0; i < 16; i++)
    {
        uint i2 = i + i;
        morton_code x = V.x;
        morton_code y = V.y;
        morton_code z = V.z;

        mcode |= (z&(0x1 << i)) << i2;
        mcode |= (y&(0x1 << i)) << ++i2;
        mcode |= (x&(0x1 << i)) << ++i2;
    }

    return mcode;
}


static __inline__ __host__ __device__ point_data ToPoint(morton_code mcode)
{
    point_data p = make_point_data(0, 0, 0);

    for (int i = 0; i < 16; i++)
    {
        p.x |= (mcode&(0x1ll << (3 * i + 2))) >> (2 * i + 2);
        p.y |= (mcode&(0x1ll << (3 * i + 1))) >> (2 * i + 1);
        p.z |= (mcode&(0x1ll << (3 * i + 0))) >> (2 * i + 0);
    }

    return p;
}


static __inline__ __host__ __device__ float3 mul3x4(float3 a, float4x4 m)
{
    return make_float3(
        a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0] + m.m[3][0],
        a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1] + m.m[3][1],
        a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2] + m.m[3][2]
    );
}


static __inline__ __host__ __device__ float4 mul4x4(float4 a, float4x4 m)
{
    return make_float4(
        a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0] + a.w * m.m[3][0],
        a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1] + a.w * m.m[3][1],
        a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2] + a.w * m.m[3][2],
        a.x * m.m[0][3] + a.y * m.m[1][3] + a.z * m.m[2][3] + a.w * m.m[3][3]
    );
}


static __inline__ __host__ __device__  float4 crs4(float3 a, float3 b, float3 c)
{
    return make_float4(
        a.z*b.y - a.y*b.z - a.z*c.y + b.z*c.y + a.y*c.z - b.y*c.z,
        -(a.z*b.x) + a.x*b.z + a.z*c.x - b.z*c.x - a.x*c.z + b.x*c.z,
        a.y*b.x - a.x*b.y - a.y*c.x + b.y*c.x + a.x*c.y - b.x*c.y,
        -(a.z*b.y*c.x) + a.y*b.z*c.x + a.z*b.x*c.y - a.x*b.z*c.y - a.y*b.x*c.z + a.x*b.y*c.z
    );
}


static __inline__ __host__ __device__  float3 crs3(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - b.y * a.z,
        a.z * b.x - b.z * a.x,
        a.x * b.y - b.x * a.y
        );
}


static __inline__ __host__ __device__ float4x4 make_float4x4(float a00, float a01, float a02, float a03,
  float a10, float a11, float a12, float a13,
  float a20, float a21, float a22, float a23,
  float a30, float a31, float a32, float a33)
{
    float4x4 a;
    a.m[0][0] = a00; a.m[0][1] = a01; a.m[0][2] = a02; a.m[0][3] = a03;
    a.m[1][0] = a10; a.m[1][1] = a11; a.m[1][2] = a12; a.m[1][3] = a13;
    a.m[2][0] = a20; a.m[2][1] = a21; a.m[2][2] = a22; a.m[2][3] = a23;
    a.m[3][0] = a30; a.m[3][1] = a31; a.m[3][2] = a32; a.m[3][3] = a33;
    return a;
}

static __inline__ __host__ __device__ void  MMmul(const float4x4& a, const float4x4& b, float4x4& c)
{
    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0] + a.m[0][3] * b.m[3][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1] + a.m[0][3] * b.m[3][1];
    c.m[0][2] = a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2] + a.m[0][3] * b.m[3][2];
    c.m[0][3] = a.m[0][0] * b.m[0][3] + a.m[0][1] * b.m[1][3] + a.m[0][2] * b.m[2][3] + a.m[0][3] * b.m[3][3];

    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0] + a.m[1][3] * b.m[3][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1] + a.m[1][3] * b.m[3][1];
    c.m[1][2] = a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2] + a.m[1][3] * b.m[3][2];
    c.m[1][3] = a.m[1][0] * b.m[0][3] + a.m[1][1] * b.m[1][3] + a.m[1][2] * b.m[2][3] + a.m[1][3] * b.m[3][3];

    c.m[2][0] = a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0] + a.m[2][3] * b.m[3][0];
    c.m[2][1] = a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1] + a.m[2][3] * b.m[3][1];
    c.m[2][2] = a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2] + a.m[2][3] * b.m[3][2];
    c.m[2][3] = a.m[2][0] * b.m[0][3] + a.m[2][1] * b.m[1][3] + a.m[2][2] * b.m[2][3] + a.m[2][3] * b.m[3][3];

    c.m[3][0] = a.m[3][0] * b.m[0][0] + a.m[3][1] * b.m[1][0] + a.m[3][2] * b.m[2][0] + a.m[3][3] * b.m[3][0];
    c.m[3][1] = a.m[3][0] * b.m[0][1] + a.m[3][1] * b.m[1][1] + a.m[3][2] * b.m[2][1] + a.m[3][3] * b.m[3][1];
    c.m[3][2] = a.m[3][0] * b.m[0][2] + a.m[3][1] * b.m[1][2] + a.m[3][2] * b.m[2][2] + a.m[3][3] * b.m[3][2];
    c.m[3][3] = a.m[3][0] * b.m[0][3] + a.m[3][1] * b.m[1][3] + a.m[3][2] * b.m[2][3] + a.m[3][3] * b.m[3][3];
}

static __inline__ __host__ __device__ float4x4 operator* (const float4x4& ma, const float4x4& mb)
{
    float4x4 mc; MMmul(ma, mb, mc); return mc;
}

static __inline__ __host__ __device__ float4x4 transpose(const float4x4& a)
{
    float4x4 b;
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            b.m[i][j] = a.m[j][i];
    return b;
}

