#include <dgemm-blocked.hpp>
#include <pmmintrin.h>
#include <immintrin.h>

const int block_small = 256;

const char *dgemm_desc = "Simple blocked dgemm Updated.";
int min(const int &a, const int &b) { return (((a) < (b)) ? (a) : (b)); }

#define min( i, j ) ( (i)<(j) ? (i): (j) )

// This is the routine to handle edge cases, needs to implement
// Maybe we can pack A and B here to increase performance
void fringe_2by2(int lda, int m, int n, int k, double *A, double *B, double* C)
{
    // This routine will do a little optimization by updating 2X2 blocks
    int r, c, p;
    int N2 = n/2*2;
    int M2 = m/2*2;
    __m128d c_00_10, c_01_11,
            a_0p_1p,
            b_p0, b_p1;

    for(r = 0; r < M2; r+=2)
    {
        for(c = 0; c < N2; c+=2)
        {
            // This is updating a 2X2 block at (r, c)
            c_00_10 = _mm_loadu_pd(C+r+c*lda);
            c_01_11 = _mm_loadu_pd(C+r+(c+1)*lda);

            // Summing up over k
            for(p = 0; p < k; p++)
            {
                a_0p_1p = _mm_loadu_pd(A+r+p*lda);
                b_p0 = _mm_load1_pd(B+p+c*lda);
                b_p1 = _mm_load1_pd(B+p+(c+1)*lda);

                c_00_10 = _mm_add_pd(c_00_10, _mm_mul_pd(a_0p_1p, b_p0));
                c_01_11 = _mm_add_pd(c_01_11, _mm_mul_pd(a_0p_1p, b_p1));
            }

            _mm_storeu_pd(C+r+c*lda, c_00_10);
            _mm_storeu_pd(C+r+(c+1)*lda, c_01_11);
        }

        if (n % 2)
        {
            // We have to update the last column by updating 2 rows each time
            c_00_10 = _mm_loadu_pd(C+r+(n-1)*lda);
            // Summing up over k
            for(p = 0; p < k; p++)
            {
                a_0p_1p = _mm_loadu_pd(A+r+p*lda);
                b_p0 = _mm_load1_pd(B+p+c*lda);

                c_00_10 = _mm_add_pd(c_00_10, _mm_mul_pd(a_0p_1p, b_p0));
            }

            _mm_storeu_pd(C+r+(n-1)*lda, c_00_10);
        }
    }

    if (m % 2)
    {
        double c_0, c_1, a, b_0, b_1;
        // Now we only have the last row to update, unroll the loop by 2
        for(c=0; c<N2; c+=2)
        {
            c_0 = C[m-1+c*lda];
            c_1 = C[m-1+(c+1)*lda];
            for(p=0; p<k; p++)
            {
                a = A[m-1+p*lda];
                b_0 = B[p+c*lda];
                b_1 = B[p+(c+1)*lda];
                c_0 += a * b_0;
                c_1 += a * b_1;
            }
            C[m-1+c*lda] = c_0;
            C[m-1+(c+1)*lda] = c_1;
        }

        {
            // There is one last element to update
            c_0 = C[m-1+(n-1)*lda];
            for(p=0; p<k; p++)
            {
                a = A[m-1+p*lda];
                b_0 = B[p+(n-1)*lda];
                c_0 += a * b_0;
            }
            C[m-1+(n-1)*lda] = c_0;
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
 * i lda k
 */
void do_block(int lda, int m, int n, int k, double *A, double *B, double *C)
{
    int i, j;
    int N4 = n/4*4;
    int M8 = m/8*8;
    double AA[m*k], BB[N4*k];
    for(j=0; j < N4; j+=4)
    {        /* Loop over the columns of C, unrolled by 4 */
            double  *src_b0 = B+j*lda,
                    *src_b1 = src_b0 + lda,
                    *src_b2 = src_b1 + lda,
                    *src_b3 = src_b2 + lda;
            double *dst_b = &BB[j*k];

            // B is packed in row-major form
            for(int i=0; i<k; i++)
            {
                *dst_b++ = *src_b0++;
                *dst_b++ = *src_b1++;
                *dst_b++ = *src_b2++;
                *dst_b++ = *src_b3++;
            }
        for (i=0; i<M8; i+=8)
        {        /* Loop over the rows of C, unrolled by 8 */
            if (j == 0) {
                    double *copy = &AA[i*k];
                    for(int j = 0; j<k; j++)
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
                int p;
                double *A = &AA[i*k];
                double* B = &BB[j*k];
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


                for(p=0; p<k; p++)
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
        fringe_2by2(lda, M8, n-N4, k, A, B+lda*N4, C+lda*N4);
        // Need to update a bottom row block and a tailing column block
        fringe_2by2(lda, m-M8, n, k, A+M8, B, C+M8);
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * */
void square_dgemm_blocked(int lda, double *A, double *B, double *C)
{
    int i, M, k, K;
    for (k=0; k<lda; k+=block_small)
    {
        // Column blocks of A
        K = min( lda-k, block_small );
        for (i=0; i<lda; i+=block_small)
        {
            // Row blocks of C
            M = min(lda-i, block_small);
            do_block(lda, M, lda, K, A+i+k*lda, B+k, C+i);
        }
    }
}
