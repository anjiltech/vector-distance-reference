/**
 * © Copyright IBM Corporation 2024. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if defined(__powerpc__)

#include <altivec.h>   /* Required for the Power GCC built-ins  */

#include "innerproduct.h"

#include <cmath>

#define FLOAT_VEC_SIZE 4
#define INT32_VEC_SIZE 4
#define INT8_VEC_SIZE  16

namespace powerpc {
    
float
fvec_inner_product_ref_ppc(const float *x, const float *y, size_t d)
{
    size_t i, base;
    float res = 0.0f;

    if (d < 32)
    {
        size_t nvec = d / FLOAT_VEC_SIZE;
        vector float *vx = (vector float *)x;
        vector float *vy = (vector float *)y;
        vector float vres = {0, 0, 0, 0};

        for (i = 0; i < nvec; ++i)
            vres += vx[i] * vy[i];

        for (i = nvec * FLOAT_VEC_SIZE; i < d; ++i)
            res += x[i] * y[i];

        return res + vres[0] + vres[1] + vres[2] + vres[3];
    }
    else if (d < 64)
    {
        base = (d / (FLOAT_VEC_SIZE * 4)) * (FLOAT_VEC_SIZE * 4);
        vector float vres0 = {0, 0, 0, 0}, vres1 = {0, 0, 0, 0},
                        vres2 = {0, 0, 0, 0}, vres3 = {0, 0, 0, 0};

        for (i = 0; i < base; i += FLOAT_VEC_SIZE * 4)
        {
            vector float *vx = (vector float *)&x[i];
            vector float *vy = (vector float *)&y[i];
            vres0 += vx[0] * vy[0];
            vres1 += vx[1] * vy[1];
            vres2 += vx[2] * vy[2];
            vres3 += vx[3] * vy[3];
        }

        for (i = base; i < d; ++i)
            res += x[i] * y[i];

        vres0 += vres1;
        vres2 += vres3;
        vres0 += vres2;
        return res + (vres0[0] + vres0[1] + vres0[2] + vres0[3]);
    }
    else if (d < 128)
    {
        base = (d / (FLOAT_VEC_SIZE * 8)) * (FLOAT_VEC_SIZE * 8);
        vector float vres0 = {0, 0, 0, 0}, vres1 = {0, 0, 0, 0},
                        vres2 = {0, 0, 0, 0}, vres3 = {0, 0, 0, 0},
                        vres4 = {0, 0, 0, 0}, vres5 = {0, 0, 0, 0},
                        vres6 = {0, 0, 0, 0}, vres7 = {0, 0, 0, 0};

        for (i = 0; i < base; i += FLOAT_VEC_SIZE * 8)
        {
            vector float *vx = (vector float *)&x[i];
            vector float *vy = (vector float *)&y[i];
            vres0 += vx[0] * vy[0];
            vres1 += vx[1] * vy[1];
            vres2 += vx[2] * vy[2];
            vres3 += vx[3] * vy[3];
            vres4 += vx[4] * vy[4];
            vres5 += vx[5] * vy[5];
            vres6 += vx[6] * vy[6];
            vres7 += vx[7] * vy[7];
        }

        for (i = base; i < d; ++i)
            res += x[i] * y[i];

        vres0 += vres1;
        vres0 += vres2;
        vres0 += vres3;
        vres0 += vres4;
        vres0 += vres5;
        vres0 += vres6;
        vres0 += vres7;
        return res + (vres0[0] + vres0[1] + vres0[2] + vres0[3]);
    }
    else
    {
        base = (d / (FLOAT_VEC_SIZE * 16)) * (FLOAT_VEC_SIZE * 16);
        vector float vres0 = {0, 0, 0, 0}, vres1 = {0, 0, 0, 0},
                        vres2 = {0, 0, 0, 0}, vres3 = {0, 0, 0, 0},
                        vres4 = {0, 0, 0, 0}, vres5 = {0, 0, 0, 0},
                        vres6 = {0, 0, 0, 0}, vres7 = {0, 0, 0, 0},
                        vres8 = {0, 0, 0, 0}, vres9 = {0, 0, 0, 0},
                        vres10 = {0, 0, 0, 0}, vres11 = {0, 0, 0, 0},
                        vres12 = {0, 0, 0, 0}, vres13 = {0, 0, 0, 0},
                        vres14 = {0, 0, 0, 0}, vres15 = {0, 0, 0, 0};

        for (i = 0; i < base; i += FLOAT_VEC_SIZE * 16)
        {
            vector float *vx = (vector float *)&x[i];
            vector float *vy = (vector float *)&y[i];
            vres0 += vx[0] * vy[0];
            vres1 += vx[1] * vy[1];
            vres2 += vx[2] * vy[2];
            vres3 += vx[3] * vy[3];
            vres4 += vx[4] * vy[4];
            vres5 += vx[5] * vy[5];
            vres6 += vx[6] * vy[6];
            vres7 += vx[7] * vy[7];
            vres8 += vx[8] * vy[8];
            vres9 += vx[9] * vy[9];
            vres10 += vx[10] * vy[10];
            vres11 += vx[11] * vy[11];
            vres12 += vx[12] * vy[12];
            vres13 += vx[13] * vy[13];
            vres14 += vx[14] * vy[14];
            vres15 += vx[15] * vy[15];
        }

        for (i = base; i < d; ++i)
            res += x[i] * y[i];

        vres0 += vres1;
        vres0 += vres2;
        vres0 += vres3;
        vres0 += vres4;
        vres0 += vres5;
        vres0 += vres6;
        vres0 += vres7;
        vres0 += vres8;
        vres0 += vres9;
        vres0 += vres10;
        vres0 += vres11;
        vres0 += vres12;
        vres0 += vres13;
        vres0 += vres14;
        vres0 += vres15;
        return res + (vres0[0] + vres0[1] + vres0[2] + vres0[3]);
    }
}
// float
// fvec_inner_product_ref_ppc(const float* x, const float* y, size_t d) {
//     size_t i;
//     float res = 0;
//     /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
//        Original code:

//        for (i = 0; i < d; i++) {
//            res += x[i] * y[i];
//        }
//        return res;
//     */
//     /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  If the input
//        array size is not a power of FLOAT_VEC_SIZE, do the remaining elements
//        in scalar mode.  */
//     size_t base;

//     vector float *vx, *vy;
//     vector float vres = {0, 0, 0, 0};

//     base = (d / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;

//     for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
//         vx = (vector float *)(&x[i]);
//         vy = (vector float *)(&y[i]);

//         vres += vx[0] * vy[0];
//     }

//     /* Handle any remaining data elements */
//     for (i = base; i < d; i++) {
//         res += x[i] * y[i];
//     }
//     return res + vres[0] + vres[1] + vres[2] + vres[3];
// }

void
fvec_inner_product_batch_4_ref_ppc(const float* __restrict x,
                                   const float* __restrict y0,
                                   const float* __restrict y1,
                                   const float* __restrict y2,
                                   const float* __restrict y3,
                                   const size_t d, float& dis0, float& dis1,
                                   float& dis2, float& dis3) {
    /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
       Original code:

       float d0 = 0;
       float d1 = 0;
       float d2 = 0;
       float d3 = 0;
       for (size_t i = 0; i < d; ++i) {
           d0 += x[i] * y0[i];
           d1 += x[i] * y1[i];
           d2 += x[i] * y2[i];
           d3 += x[i] * y3[i];
       }

       dis0 = d0;
       dis1 = d1;
       dis2 = d2;
       dis3 = d3;
   */
    /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  If the input
       array size is not a power of FLOAT_VEC_SIZE, do the remaining elements
       in scalar mode.  */

    size_t base, remainder;
    vector float *vx, *vy0, *vy1, *vy2, *vy3;
    vector float vd0 = {0.0, 0.0, 0.0, 0.0};
    vector float vd1 = {0.0, 0.0, 0.0, 0.0};
    vector float vd2 = {0.0, 0.0, 0.0, 0.0};
    vector float vd3 = {0.0, 0.0, 0.0, 0.0};

    base = (d / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;
    remainder = d % FLOAT_VEC_SIZE;

    for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
        vx = (vector float *)(&x[i]);
        vy0 = (vector float *)(&y0[i]);
        vy1 = (vector float *)(&y1[i]);
        vy2 = (vector float *)(&y2[i]);
        vy3 = (vector float *)(&y3[i]);

        vd0 += vx[0] * vy0[0];
        vd1 += vx[0] * vy1[0];
        vd2 += vx[0] * vy2[0];
        vd3 += vx[0] * vy3[0];
    }

    dis0 = vd0[0] + vd0[1] + vd0[2] + vd0[3];
    dis1 = vd1[0] + vd1[1] + vd1[2] + vd1[3];
    dis2 = vd2[0] + vd2[1] + vd2[2] + vd2[3];
    dis3 = vd3[0] + vd3[1] + vd3[2] + vd3[3];

    /* Handle any remaining data elements */
    if (remainder != 0) {
        float d0 = 0;
        float d1 = 0;
        float d2 = 0;
        float d3 = 0;

        for (size_t i = base; i < d; i++) {
            d0 += x[i] * y0[i];
            d1 += x[i] * y1[i];
            d2 += x[i] * y2[i];
            d3 += x[i] * y3[i];
        }

        dis0 += d0;
        dis1 += d1;
        dis2 += d2;
        dis3 += d3;
    }
}

int32_t
ivec_inner_product_ref_ppc(const int8_t* x, const int8_t* y, size_t d) {
    size_t i;
    int32_t res = 0;

    /* Attempts to mannually vectorize and manually unroll the loop
       do not seem to improve the performance. */
    for (i = 0; i < d; i++) {
        res += (int32_t)x[i] * y[i];
    }
    return res;
}

} // namespace powerpc 

#endif
