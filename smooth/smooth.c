#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include <immintrin.h>

/* 
 * Please fill in the following team struct 
 */
who_t who = {
    "bruhhhhbbbbb",           /* Scoreboard name */

    "Scott Hong",      /* First member full name */
    "jh4ctf@virginia.edu",     /* First member email address */
};

/*** UTILITY FUNCTIONS ***/

/* You are free to use these utility functions, or write your own versions
 * of them. */

/* A struct used to compute averaged pixel value */
typedef struct {
    unsigned short red;
    unsigned short green;
    unsigned short blue;
    unsigned short alpha;
    unsigned short num;
} pixel_sum;

/* Compute min and max of two integers, respectively */
static int min(int a, int b) { return (a < b ? a : b); }
static int max(int a, int b) { return (a > b ? a : b); }

/* 
 * initialize_pixel_sum - Initializes all fields of sum to 0 
 */
static void initialize_pixel_sum(pixel_sum *sum) 
{
    sum->red = sum->green = sum->blue = sum->alpha = 0;
    sum->num = 0;
    return;
}

/* 
 * accumulate_sum - Accumulates field values of p in corresponding 
 * fields of sum 
 */
static void accumulate_sum(pixel_sum *sum, pixel p) 
{
    sum->red += (int) p.red;
    sum->green += (int) p.green;
    sum->blue += (int) p.blue;
    sum->alpha += (int) p.alpha;
    sum->num++;
    return;
}

/* 
 * assign_sum_to_pixel - Computes averaged pixel value in current_pixel 
 */
static void assign_sum_to_pixel(pixel *current_pixel, pixel_sum sum) 
{
    current_pixel->red = (unsigned short) (sum.red/sum.num);
    current_pixel->green = (unsigned short) (sum.green/sum.num);
    current_pixel->blue = (unsigned short) (sum.blue/sum.num);
    current_pixel->alpha = (unsigned short) (sum.alpha/sum.num);
    return;
}

/* 
 * avg - Returns averaged pixel value at (i,j) 
 */
static pixel avg(int dim, int i, int j, pixel *src) 
{
    pixel_sum sum;
    pixel current_pixel;

    initialize_pixel_sum(&sum);
    for(int jj=max(j-1, 0); jj <= min(j+1, dim-1); jj++) 
	for(int ii=max(i-1, 0); ii <= min(i+1, dim-1); ii++) 
	    accumulate_sum(&sum, src[RIDX(ii,jj,dim)]);

    assign_sum_to_pixel(&current_pixel, sum);
 
    return current_pixel;
}



/******************************************************
 * Your different versions of the smooth go here
 ******************************************************/

/* 
 * naive_smooth - The naive baseline version of smooth
 */
char naive_smooth_descr[] = "naive_smooth: Naive baseline implementation";
void naive_smooth(int dim, pixel *src, pixel *dst) 
{
    for (int i = 0; i < dim; i++)
	for (int j = 0; j < dim; j++)
            dst[RIDX(i,j, dim)] = avg(dim, i, j, src);
}
/* 
 * smooth - Your current working version of smooth
 *          Our supplied version simply calls naive_smooth
 */

char another_smooth_descr[] = "another_smooth: Another version of smooth";
void another_smooth(int dim, pixel *src, pixel *dst) 
{
    // for (int i = 1; i < dim-1; i++){
    //     for (int j = 1; j < dim-1; j++){
    //         dst[RIDX(i,j, dim)] = avg(dim, i, j, src);
    //     }
    // }

    // Process of add middle section
    for (int j = 1; j <= dim - 2; j ++)
    {
        for (int i = 1; i <= dim - 2; i+=4)
        {
            /*
            int *x = &array[i * N + j];
            ...
            int *y = x + N; --- instead of &array[(i+1) * N + j] 
            */
            unsigned short curr_pixel[16];
            int pointer = j * dim + i;
            pixel final_p_1,final_p_2,final_p_3,final_p_4;
            __m128i p_1 = _mm_loadu_si128((__m128i *)&src[ pointer - dim - 1]);
            __m256i p_1_r = _mm256_cvtepu8_epi16(p_1);
            __m128i p_2 = _mm_loadu_si128((__m128i *)&src[ pointer - dim]);
            __m256i p_2_r = _mm256_cvtepu8_epi16(p_2);
            __m128i p_3 = _mm_loadu_si128((__m128i *)&src[ pointer - dim + 1]);
            __m256i p_3_r = _mm256_cvtepu8_epi16(p_3);
            __m128i p_4 = _mm_loadu_si128((__m128i *)&src[ pointer - 1]);
            __m256i p_4_r = _mm256_cvtepu8_epi16(p_4);
            __m128i p_5 = _mm_loadu_si128((__m128i *)&src[ pointer]);
            __m256i p_5_r = _mm256_cvtepu8_epi16(p_5);
            __m128i p_6 = _mm_loadu_si128((__m128i *)&src[ pointer + 1]);
            __m256i p_6_r = _mm256_cvtepu8_epi16(p_6);
            __m128i p_7 = _mm_loadu_si128((__m128i *)&src[ pointer + dim - 1]);
            __m256i p_7_r = _mm256_cvtepu8_epi16(p_7);
            __m128i p_8 = _mm_loadu_si128((__m128i *)&src[ pointer + dim]);
            __m256i p_8_r = _mm256_cvtepu8_epi16(p_8);
            __m128i p_9 = _mm_loadu_si128((__m128i *)&src[ pointer + dim + 1]);
            __m256i p_9_r = _mm256_cvtepu8_epi16(p_9);


            //Sum of all pixels 
            __m256i sum = p_1_r;
            sum = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16( _mm256_add_epi16( _mm256_add_epi16( _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(sum, p_2_r), p_3_r), p_4_r), p_5_r), p_6_r), p_7_r), p_8_r), p_9_r);
            // Store the sum in a place holder
            _mm256_storeu_si256((__m256i *)curr_pixel, sum);

            final_p_1.red = (unsigned short)(curr_pixel[0] * 7282 >> 16);
            final_p_1.green = (unsigned short)(curr_pixel[1] * 7282 >> 16);
            final_p_1.blue = (unsigned short)(curr_pixel[2] * 7282 >> 16);
            final_p_1.alpha = (unsigned short)(curr_pixel[3] * 7282 >> 16);
            

            final_p_2.red = (unsigned short)(curr_pixel[4] * 7282 >> 16);
            final_p_2.green = (unsigned short)(curr_pixel[5] * 7282 >> 16);
            final_p_2.blue = (unsigned short)(curr_pixel[6] * 7282 >> 16);
            final_p_2.alpha = (unsigned short)(curr_pixel[7] * 7282 >> 16);
            

            final_p_3.red = (unsigned short)(curr_pixel[8] * 7282 >> 16);
            final_p_3.green = (unsigned short)(curr_pixel[9] * 7282 >> 16);
            final_p_3.blue = (unsigned short)(curr_pixel[10] * 7282 >> 16);
            final_p_3.alpha = (unsigned short)(curr_pixel[11] * 7282 >> 16);
            

            final_p_4.red = (unsigned short)(curr_pixel[12] * 7282 >> 16);
            final_p_4.green = (unsigned short)(curr_pixel[13] * 7282 >> 16);
            final_p_4.blue = (unsigned short)(curr_pixel[14] * 7282 >> 16);
            final_p_4.alpha = (unsigned short)(curr_pixel[15] * 7282 >> 16);
            
            dst[pointer] = final_p_1;
            dst[pointer + 1] = final_p_2;
            dst[pointer + 2] = final_p_3;
            dst[pointer + 3] = final_p_4;
        }
    }
    //Sum of Upper Left Corner
    unsigned short upper_left_corner_r = (src[RIDX(0, 0, dim)].red + src[RIDX(1, 1, dim)].red + src[RIDX(1, 0, dim)].red + src[RIDX(0, 1, dim)].red );
    dst[RIDX(0, 0, dim)].red = upper_left_corner_r>>2;
    unsigned short upper_left_corner_g = (src[RIDX(0, 0, dim)].green + src[RIDX(1, 1, dim)].green+ src[RIDX(1, 0, dim)].green + src[RIDX(0, 1, dim)].green) ;
    dst[RIDX(0, 0, dim)].green = upper_left_corner_g>>2;
    unsigned short upper_left_corner_b= (src[RIDX(0, 0, dim)].blue  + src[RIDX(1, 1, dim)].blue + src[RIDX(1, 0, dim)].blue + src[RIDX(0, 1, dim)].blue  );
    dst[RIDX(0, 0, dim)].blue = upper_left_corner_b>>2;
    unsigned short upper_left_corner_a = (src[RIDX(0, 0, dim)].alpha + src[RIDX(1, 1, dim)].alpha + src[RIDX(1, 0, dim)].alpha + src[RIDX(0, 1, dim)].alpha );
    dst[RIDX(0, 0, dim)].alpha = upper_left_corner_a>>2;
    //Sum of Upper Right Corner
    unsigned short upper_right_corner_r = (src[RIDX(0, dim-2, dim)].red + src[RIDX(1, dim-2, dim)].red + src[RIDX(1, dim-1, dim)].red + src[RIDX(0, dim-1, dim)].red );
    dst[RIDX(0, dim-1, dim)].red = upper_right_corner_r>>2;
    unsigned short upper_right_corner_g = (src[RIDX(0, dim-2, dim)].green + src[RIDX(1, dim-2, dim)].green + src[RIDX(1, dim-1, dim)].green + src[RIDX(0, dim-1, dim)].green );
    dst[RIDX(0, dim-1, dim)].green = upper_right_corner_g>>2;
    unsigned short upper_right_corner_b= (src[RIDX(0, dim-2, dim)].blue + src[RIDX(1, dim-2, dim)].blue + src[RIDX(1, dim-1, dim)].blue + src[RIDX(0, dim-1, dim)].blue );
    dst[RIDX(0, dim-1, dim)].blue = upper_right_corner_b>>2;
    unsigned short upper_right_corner_a = (src[RIDX(0, dim-2, dim)].alpha + src[RIDX(1, dim-2, dim)].alpha + src[RIDX(1, dim-1, dim)].alpha + src[RIDX(0, dim-1, dim)].alpha );
    dst[RIDX(0, dim-1, dim)].alpha = upper_right_corner_a>>2;
    //Sum of Bottom Left Corner
    unsigned short bottom_left_corner_r = (src[RIDX(dim-2, 0, dim)].red + src[RIDX(dim-2, 1, dim)].red + src[RIDX(dim-1, 0, dim)].red + src[RIDX(dim-1, 1, dim)].red );
    dst[RIDX(dim-1, 0, dim)].red = bottom_left_corner_r>>2;
    unsigned short bottom_left_corner_g = (src[RIDX(dim-2, 0, dim)].green + src[RIDX(dim-2, 1, dim)].green + src[RIDX(dim-1, 0, dim)].green + src[RIDX(dim-1, 1, dim)].green );
    dst[RIDX(dim-1, 0, dim)].green = bottom_left_corner_g>>2;
    unsigned short bottom_left_corner_b= (src[RIDX(dim-2, 0, dim)].blue + src[RIDX(dim-2, 1, dim)].blue + src[RIDX(dim-1, 0, dim)].blue + src[RIDX(dim-1, 1, dim)].blue );
    dst[RIDX(dim-1, 0, dim)].blue = bottom_left_corner_b>>2;
    unsigned short bottom_left_corner_a = (src[RIDX(dim-2, 0, dim)].alpha + src[RIDX(dim-2, 1, dim)].alpha + src[RIDX(dim-1, 0, dim)].alpha + src[RIDX(dim-1, 1, dim)].alpha );
    dst[RIDX(dim-1, 0, dim)].alpha = bottom_left_corner_a>>2;
    //Sum of Bottom Right Corner
    unsigned short bottom_right_corner_r = (src[RIDX(dim-2, dim-2, dim)].red + src[RIDX(dim-2, dim-1, dim)].red + src[RIDX(dim-1, dim-2, dim)].red + src[RIDX(dim-1, dim-1, dim)].red );
    dst[RIDX(dim-1,dim-1, dim)].red = bottom_right_corner_r>>2;
    unsigned short bottom_right_corner_g = (src[RIDX(dim-2, dim-2, dim)].green + src[RIDX(dim-2, dim-1, dim)].green + src[RIDX(dim-1, dim-2, dim)].green + src[RIDX(dim-1, dim-1, dim)].green );
    dst[RIDX(dim-1,dim-1, dim)].green = bottom_right_corner_g>>2;
    unsigned short bottom_right_corner_b= (src[RIDX(dim-2, dim-2, dim)].blue + src[RIDX(dim-2, dim-1, dim)].blue + src[RIDX(dim-1, dim-2, dim)].blue + src[RIDX(dim-1, dim-1, dim)].blue );
    dst[RIDX(dim-1,dim-1, dim)].blue = bottom_right_corner_b>>2;
    unsigned short bottom_right_corner_a = (src[RIDX(dim-2, dim-2, dim)].alpha + src[RIDX(dim-2, dim-1, dim)].alpha + src[RIDX(dim-1, dim-2, dim)].alpha + src[RIDX(dim-1, dim-1, dim)].alpha );
    dst[RIDX(dim-1,dim-1, dim)].alpha = bottom_right_corner_a>>2;

    //Sum of Top Border 
    for (int counter = 1; counter <dim-1;counter++){
        unsigned short t_r = (src[RIDX(0, counter-1, dim)].red + src[RIDX(0, counter, dim)].red + src[RIDX(0, counter+1, dim)].red + src[RIDX(1, counter-1, dim)].red+ src[RIDX(1, counter, dim)].red+ src[RIDX(1, counter+1, dim)].red ) ;
        dst[RIDX(0, counter, dim)].red = t_r/6;
        unsigned short t_g = (src[RIDX(0, counter-1, dim)].green + src[RIDX(0, counter, dim)].green + src[RIDX(0, counter+1, dim)].green + src[RIDX(1, counter-1, dim)].green+ src[RIDX(1, counter, dim)].green+ src[RIDX(1, counter+1, dim)].green ) ;
        dst[RIDX(0, counter, dim)].green = t_g/6;
        unsigned short t_b = (src[RIDX(0, counter-1, dim)].blue + src[RIDX(0, counter, dim)].blue + src[RIDX(0, counter+1, dim)].blue + src[RIDX(1, counter-1, dim)].blue+ src[RIDX(1, counter, dim)].blue+ src[RIDX(1, counter+1, dim)].blue ) ;
        dst[RIDX(0, counter, dim)].blue = t_b/6;
        unsigned short t_a = (src[RIDX(0, counter-1, dim)].alpha + src[RIDX(0, counter, dim)].alpha + src[RIDX(0, counter+1, dim)].alpha + src[RIDX(1, counter-1, dim)].alpha+ src[RIDX(1, counter, dim)].alpha+ src[RIDX(1, counter+1, dim)].alpha ) ;
        dst[RIDX(0, counter, dim)].alpha = t_a/6;
    }
    //Sum of Right Border 
    for (int counter = 1; counter <dim-1;counter++){
        unsigned short r_r = (src[RIDX(counter-1, dim-2, dim)].red + src[RIDX(counter, dim-2, dim)].red + src[RIDX(counter+1, dim-2, dim)].red + src[RIDX(counter-1, dim-1, dim)].red + src[RIDX(counter, dim-1, dim)].red + src[RIDX(counter+1, dim-1, dim)].red ) ;
        dst[RIDX(counter, dim-1, dim)].red = r_r/6;
        unsigned short r_g = (src[RIDX(counter-1, dim-2, dim)].green + src[RIDX(counter, dim-2, dim)].green + src[RIDX(counter+1, dim-2, dim)].green + src[RIDX(counter-1, dim-1, dim)].green + src[RIDX(counter, dim-1, dim)].green + src[RIDX(counter+1, dim-1, dim)].green ) ;
        dst[RIDX(counter, dim-1, dim)].green = r_g/6;
        unsigned short r_b = (src[RIDX(counter-1, dim-2, dim)].blue + src[RIDX(counter, dim-2, dim)].blue + src[RIDX(counter+1, dim-2, dim)].blue + src[RIDX(counter-1, dim-1, dim)].blue + src[RIDX(counter, dim-1, dim)].blue + src[RIDX(counter+1, dim-1, dim)].blue ) ;
        dst[RIDX(counter, dim-1, dim)].blue = r_b/6;
        unsigned short r_a = (src[RIDX(counter-1, dim-2, dim)].alpha + src[RIDX(counter, dim-2, dim)].alpha + src[RIDX(counter+1, dim-2, dim)].alpha + src[RIDX(counter-1, dim-1, dim)].alpha + src[RIDX(counter, dim-1, dim)].alpha + src[RIDX(counter+1, dim-1, dim)].alpha ) ;
        dst[RIDX(counter, dim-1, dim)].alpha = r_a/6;
    }

    //Sum of Left Border 
    for (int counter = 1; counter <dim-1;counter++){
        unsigned short l_r = (src[RIDX(counter-1, 0, dim)].red + src[RIDX(counter, 0, dim)].red + src[RIDX(counter+1, 0, dim)].red + src[RIDX(counter-1, 1, dim)].red + src[RIDX(counter, 1, dim)].red + src[RIDX(counter+1, 1, dim)].red ) ;
        dst[RIDX(counter, 0, dim)].red = l_r/6;
        unsigned short l_g = (src[RIDX(counter-1, 0, dim)].green + src[RIDX(counter, 0, dim)].green + src[RIDX(counter+1, 0, dim)].green + src[RIDX(counter-1, 1, dim)].green + src[RIDX(counter, 1, dim)].green + src[RIDX(counter+1, 1, dim)].green ) ;
        dst[RIDX(counter, 0, dim)].green = l_g/6;
        unsigned short l_b = (src[RIDX(counter-1, 0, dim)].blue + src[RIDX(counter, 0, dim)].blue + src[RIDX(counter+1, 0, dim)].blue + src[RIDX(counter-1, 1, dim)].blue + src[RIDX(counter, 1, dim)].blue + src[RIDX(counter+1, 1, dim)].blue ) ;
        dst[RIDX(counter, 0, dim)].blue = l_b/6;
        unsigned short l_a = (src[RIDX(counter-1, 0, dim)].alpha + src[RIDX(counter, 0, dim)].alpha + src[RIDX(counter+1, 0, dim)].alpha + src[RIDX(counter-1, 1, dim)].alpha + src[RIDX(counter, 1, dim)].alpha + src[RIDX(counter+1, 1, dim)].alpha ) ;
        dst[RIDX(counter, 0, dim)].alpha = l_a/6;
    }
    //Sum of Bottom Border 
    for (int counter = 1; counter <dim-1;counter++){
        unsigned short b_r = (src[RIDX(dim-2, counter-1, dim)].red + src[RIDX(dim-2, counter, dim)].red + src[RIDX(dim-2, counter+1, dim)].red + src[RIDX(dim-1, counter-1, dim)].red+ src[RIDX(dim-1, counter, dim)].red+ src[RIDX(dim-1, counter+1, dim)].red ) ;
        dst[RIDX(dim-1, counter, dim)].red = b_r/6;
        unsigned short b_g = (src[RIDX(dim-2, counter-1, dim)].green + src[RIDX(dim-2, counter, dim)].green + src[RIDX(dim-2, counter+1, dim)].green + src[RIDX(dim-1, counter-1, dim)].green+ src[RIDX(dim-1, counter, dim)].green+ src[RIDX(dim-1, counter+1, dim)].green ) ;
        dst[RIDX(dim-1, counter, dim)].green = b_g/6;
        unsigned short b_b = (src[RIDX(dim-2, counter-1, dim)].blue + src[RIDX(dim-2, counter, dim)].blue + src[RIDX(dim-2, counter+1, dim)].blue + src[RIDX(dim-1, counter-1, dim)].blue+ src[RIDX(dim-1, counter, dim)].blue+ src[RIDX(dim-1, counter+1, dim)].blue ) ;
        dst[RIDX(dim-1, counter, dim)].blue = b_b/6;
        unsigned short b_a = (src[RIDX(dim-2, counter-1, dim)].alpha + src[RIDX(dim-2, counter, dim)].alpha + src[RIDX(dim-2, counter+1, dim)].alpha + src[RIDX(dim-1, counter-1, dim)].alpha+ src[RIDX(dim-1, counter, dim)].alpha+ src[RIDX(dim-1, counter+1, dim)].alpha ) ;
        dst[RIDX(dim-1, counter, dim)].alpha = b_a/6;
    }
	
}

/*********************************************************************
 * register_smooth_functions - Register all of your different versions
 *     of the smooth function by calling the add_smooth_function() for
 *     each test function. When you run the benchmark program, it will
 *     test and report the performance of each registered test
 *     function.  
 *********************************************************************/

void register_smooth_functions() {
    add_smooth_function(&naive_smooth, naive_smooth_descr);
    add_smooth_function(&another_smooth, another_smooth_descr);
}
