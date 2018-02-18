#include <dgemm-blocked.hpp>
#include <pmmintrin.h>
#include <immintrin.h>

const int block_small = 256;

const char *dgemm_desc = "Simple blocked dgemm Updated.";
int min(const int &a, const int &b) { return (((a) < (b)) ? (a) : (b)); }

#define min( i, j ) ( (i)<(j) ? (i): (j) )

// Use this to clean the fringes and overflow
// Changing to 8 by 4 made the fringe blocks very slow due to larger block size
// So this initial code runs smaller two by two blocks using intrinsics for some improvement
// Then handles cases of overflow where M and N is not divisible by 2
void fringe(int lda, int M, int N, int K, double *A, double *B, double* C)
{
    int i, j, k;
    int N2 = N/2*2;
    int M2 = M/2*2;
    __m128d c_00_10, c_01_11,
            a01, b0, b1;

    for(i = 0; i < M2; i+=2)
    {
        for(j = 0; j < N2; j+=2)
        {
            c_00_10 = _mm_loadu_pd(C+i+j*lda);
            c_01_11 = _mm_loadu_pd(C+i+(j+1)*lda);

            for(k = 0; k < K; k++)
            {
                a01 = _mm_loadu_pd(A+i+k*lda);
                b0 = _mm_loaddup_pd(B+k+j*lda);
                b1 = _mm_loaddup_pd(B+k+(j+1)*lda);

                c_00_10 = _mm_add_pd(c_00_10, _mm_mul_pd(a01, b0));
                c_01_11 = _mm_add_pd(c_01_11, _mm_mul_pd(a01, b1));
            }

            _mm_storeu_pd(C+i+j*lda, c_00_10);
            _mm_storeu_pd(C+i+(j+1)*lda, c_01_11);
        }

        // Complete the last column overflow with minor intrinsics
        if (N != N2)
        {
            c_00_10 = _mm_loadu_pd(C+i+(N-1)*lda);
            for(k = 0; k < K; k++)
            {
                a01 = _mm_loadu_pd(A+i+k*lda);
                b0 = _mm_load1_pd(B+k+j*lda);

                c_00_10 = _mm_add_pd(c_00_10, _mm_mul_pd(a01, b0));
            }

            _mm_storeu_pd(C+i+(N-1)*lda, c_00_10);
        }
    }

    // Handle rest of the overflow naively
    if (M != M2)
    {
        double c0, c1, a, b0, b1;
        for(j=0; j<N2; j+=2)
        {
            c0 = C[M-1+j*lda];
            c1 = C[M-1+(j+1)*lda];
            for(k=0; k<K; k++)
            {
                a = A[M-1+k*lda];
                b0 = B[k+j*lda];
                b1 = B[k+(j+1)*lda];
                c0 += a * b0;
                c1 += a * b1;
            }
            C[M-1+j*lda] = c0;
            C[M-1+(j+1)*lda] = c1;
        }

        {
            // There is one last element to update
            c0 = C[M-1+(N-1)*lda];
            for(k=0; k<K; k++)
            {
                a = A[M-1+k*lda];
                b0 = B[k+(N-1)*lda];
                c0 += a * b0;
            }
            C[M-1+(N-1)*lda] = c0;
        }
    }

}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 * https://felix.abecassis.me/2011/09/cpp-getting-started-with-sse/
 * https://stackoverflow.com/questions/18499971/efficient-4x4-matrix-multiplication-c-vs-assembly
 * https://www.inf.ethz.ch/personal/markusp/teaching/263-2300-ETH-spring11/slides/class17.pdf
 * http://fhtr.blogspot.com/2010/02/4x4-float-matrix-multiplication-using.html?m=1
 */
void do_block(int lda, int M, int N, int K, double *A, double *B, double *C, int M8, int N4)
{
    double AA[M*K], BB[N4*K];
    for(int j=0; j < N4; j+=4)
    {        /* Loop over the columns of C, unrolled by 4 */
            double  *src_b0 = B+j*lda,
                    *src_b1 = src_b0 + lda,
                    *src_b2 = src_b1 + lda,
                    *src_b3 = src_b2 + lda;
            double *dst_b = &BB[j*K];

            // B is packed in row-major form
            for(int i=0; i<K; i++)
            {
                *dst_b++ = *src_b0++;
                *dst_b++ = *src_b1++;
                *dst_b++ = *src_b2++;
                *dst_b++ = *src_b3++;
            }
        for (int i=0; i<M8; i+=8)
        {        /* Loop over the rows of C, unrolled by 8 */
            if (j == 0) {
                    double *copy = &AA[i*K];
                    for(int j = 0; j<K; j++)
                    {
                        double *src_a = A+i+j*lda;
                        *copy = *src_a;
                        *(copy+1) = *(src_a+1);
                        *(copy+2) = *(src_a+2);
                        *(copy+3) = *(src_a+3);
                        *(copy+4) = *(src_a+4);
                        *(copy+5) = *(src_a+5);
                        *(copy+6) = *(src_a+6);
                        *(copy+7) = *(src_a+7);
                        copy += 8;
                    }

            }
                double *A = &AA[i*K];
                double* B = &BB[j*K];
                double *CC =  C+i+j*lda;
                __m256d c_00_30, c_01_31, c_02_32, c_03_33, c_40_70, c_41_71, c_42_72, c_43_73,
                        a_0p_3p, a_4p_7p,
                        b_p0, b_p1, b_p2, b_p3;

                c_00_30 = _mm256_loadu_pd(CC);
                c_01_31 = _mm256_loadu_pd(CC+lda);
                c_02_32 = _mm256_loadu_pd(CC+lda*2);
                c_03_33 = _mm256_loadu_pd(CC+lda*3);
                c_40_70 = _mm256_loadu_pd(CC+4);
                c_41_71 = _mm256_loadu_pd(CC+4+lda);
                c_42_72 = _mm256_loadu_pd(CC+4+lda*2);
                c_43_73 = _mm256_loadu_pd(CC+4+lda*3);


                for(int k=0; k<K; k++)
                {
                    // Load a
                    a_0p_3p = _mm256_loadu_pd(A);
                    a_4p_7p = _mm256_loadu_pd(A+4);
                    A += 8;

                    // Load b
                    b_p0 = _mm256_broadcast_sd(B);
                    b_p1 = _mm256_broadcast_sd(B+1);
                    b_p2 = _mm256_broadcast_sd(B+2);
                    b_p3 = _mm256_broadcast_sd(B+3);

                    B += 4;

                    // First four rows of C updated once
                    c_00_30 = _mm256_add_pd(c_00_30, _mm256_mul_pd(a_0p_3p, b_p0));
                    c_01_31 = _mm256_add_pd(c_01_31, _mm256_mul_pd(a_0p_3p, b_p1));
                    c_02_32 = _mm256_add_pd(c_02_32, _mm256_mul_pd(a_0p_3p, b_p2));
                    c_03_33 = _mm256_add_pd(c_03_33, _mm256_mul_pd(a_0p_3p, b_p3));

                    // Last four rows of C updated once
                    c_40_70 = _mm256_add_pd(c_40_70, _mm256_mul_pd(a_4p_7p, b_p0));
                    c_41_71 = _mm256_add_pd(c_41_71, _mm256_mul_pd(a_4p_7p, b_p1));
                    c_42_72 = _mm256_add_pd(c_42_72, _mm256_mul_pd(a_4p_7p, b_p2));
                    c_43_73 = _mm256_add_pd(c_43_73, _mm256_mul_pd(a_4p_7p, b_p3));
                }

                _mm256_storeu_pd(CC, c_00_30);
                _mm256_storeu_pd(CC+lda, c_01_31);
                _mm256_storeu_pd(CC+lda*2, c_02_32);
                _mm256_storeu_pd(CC+lda*3, c_03_33);
                _mm256_storeu_pd(CC+4, c_40_70);
                _mm256_storeu_pd(CC+4+lda, c_41_71);
                _mm256_storeu_pd(CC+4+lda*2, c_42_72);
                _mm256_storeu_pd(CC+4+lda*3, c_43_73);
        }
    }
    {
        // handle fringes with B and C matrices
        fringe(lda, M8, N-N4, K, A, B+lda*N4, C+lda*N4);
        fringe(lda, M-M8, N, K, A+M8, B, C+M8);
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * */
void square_dgemm_blocked(int lda, double *A, double *B, double *C)
{
    int N4 = lda/4*4;
    for (int k=0; k<lda; k+=block_small)
    {
        // Column blocks of A
        int K = min( lda-k, block_small );
        for (int i=0; i<lda; i+=block_small)
        {
            // Row blocks of C
            int M = min(lda-i, block_small);
            int M8 = M/8*8;
            // need to calc M8 and N4 here then pass below
            do_block(lda, M, lda, K, A+i+k*lda, B+k, C+i, M8, N4);
            // can i move fringes here?
        }
    }
}
