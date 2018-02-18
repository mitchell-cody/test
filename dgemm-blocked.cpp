#include <dgemm-blocked.hpp>
#include <pmmintrin.h>
#include <immintrin.h>

const int block_small = 256;

const char *dgemm_desc = "Simple blocked dgemm Updated.";
int min(const int &a, const int &b) { return (((a) < (b)) ? (a) : (b)); }

#define nb 1000
#define DEBUG 0

#define min( i, j ) ( (i)<(j) ? (i): (j) )

// This subroutine copy an 8 by k block of A
void copyA8(int k, double *A, int lda, double *copy)
{
    int j;
    for(j = 0; j<k; j++)
    {
        double *a_ptr = A+j*lda;
        *copy = *a_ptr;
        *(copy+1) = *(a_ptr+1);
        *(copy+2) = *(a_ptr+2);
        *(copy+3) = *(a_ptr+3);
        *(copy+4) = *(a_ptr+4);
        *(copy+5) = *(a_ptr+5);
        *(copy+6) = *(a_ptr+6);
        *(copy+7) = *(a_ptr+7);

        copy += 8;
    }
}

void copyB(int k, double *B, int ldb, double *copy)
{
    int i;
    double *b_i0 = B, *b_i1 = B+ldb, *b_i2 = B+2*ldb, *b_i3 = B+3*ldb;

    // B is packed in row-major form
    for(i=0; i<k; i++)
    {
        *copy++ = *b_i0++;
        *copy++ = *b_i1++;
        *copy++ = *b_i2++;
        *copy++ = *b_i3++;
    }
}

void copyB3(int k, double *B, int ldb, double *copy)
{
    int i;
    double *b_i0 = B, *b_i1 = B+ldb, *b_i2 = B+2*ldb;

    // B is packed in row-major form
    for(i=0; i<k; i++)
    {
        *copy++ = *b_i0++;
        *copy++ = *b_i1++;
        *copy++ = *b_i2++;
    }
}

// Test another subroutine that updates an 8 by 4 block of C at a time
void update8X4(int k, double *A, int lda, double* B, int ldb, double *C, int ldc)
{
    int p;
    __m256d c_00_30, c_01_31, c_02_32, c_03_33, c_40_70, c_41_71, c_42_72, c_43_73,
            a_0p_3p, a_4p_7p,
            b_p0, b_p1, b_p2, b_p3;

    c_00_30 = _mm256_loadu_pd(C);
    c_01_31 = _mm256_loadu_pd(C+ldc);
    c_02_32 = _mm256_loadu_pd(C+ldc*2);
    c_03_33 = _mm256_loadu_pd(C+ldc*3);
    c_40_70 = _mm256_loadu_pd(C+4);
    c_41_71 = _mm256_loadu_pd(C+4+ldc);
    c_42_72 = _mm256_loadu_pd(C+4+ldc*2);
    c_43_73 = _mm256_loadu_pd(C+4+ldc*3);


    for(p=0; p<k; p++)
    {
        // Load a
        a_0p_3p = _mm256_loadu_pd(A);
        a_4p_7p = _mm256_loadu_pd(A+4);
        A += lda;

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

    _mm256_storeu_pd(C, c_00_30);
    _mm256_storeu_pd(C+ldc, c_01_31);
    _mm256_storeu_pd(C+ldc*2, c_02_32);
    _mm256_storeu_pd(C+ldc*3, c_03_33);
    _mm256_storeu_pd(C+4, c_40_70);
    _mm256_storeu_pd(C+4+ldc, c_41_71);
    _mm256_storeu_pd(C+4+ldc*2, c_42_72);
    _mm256_storeu_pd(C+4+ldc*3, c_43_73);
}

// Test another subroutine that updates an 8 by 4 block of C at a time
void update8X3(int k, double *A, int lda, double* B, int ldb, double *C, int ldc)
{
    int p;
    __m256d c_00_30, c_01_31, c_02_32, c_40_70, c_41_71, c_42_72,
            a_0p_3p, a_4p_7p,
            b_p0, b_p1, b_p2;

    c_00_30 = _mm256_loadu_pd(C);
    c_01_31 = _mm256_loadu_pd(C+ldc);
    c_02_32 = _mm256_loadu_pd(C+ldc*2);
    c_40_70 = _mm256_loadu_pd(C+4);
    c_41_71 = _mm256_loadu_pd(C+4+ldc);
    c_42_72 = _mm256_loadu_pd(C+4+ldc*2);


    for(p=0; p<k; p++)
    {
        // Load a
        a_0p_3p = _mm256_loadu_pd(A);
        a_4p_7p = _mm256_loadu_pd(A+4);
        A += lda;

        // Load b
        b_p0 = _mm256_broadcast_sd(B);
        b_p1 = _mm256_broadcast_sd(B+1);
        b_p2 = _mm256_broadcast_sd(B+2);

        B += 3;

        // First four rows of C updated once
        c_00_30 = _mm256_add_pd(c_00_30, _mm256_mul_pd(a_0p_3p, b_p0));
        c_01_31 = _mm256_add_pd(c_01_31, _mm256_mul_pd(a_0p_3p, b_p1));
        c_02_32 = _mm256_add_pd(c_02_32, _mm256_mul_pd(a_0p_3p, b_p2));

        // Last four rows of C updated once
        c_40_70 = _mm256_add_pd(c_40_70, _mm256_mul_pd(a_4p_7p, b_p0));
        c_41_71 = _mm256_add_pd(c_41_71, _mm256_mul_pd(a_4p_7p, b_p1));
        c_42_72 = _mm256_add_pd(c_42_72, _mm256_mul_pd(a_4p_7p, b_p2));
    }

    _mm256_storeu_pd(C, c_00_30);
    _mm256_storeu_pd(C+ldc, c_01_31);
    _mm256_storeu_pd(C+ldc*2, c_02_32);
    _mm256_storeu_pd(C+4, c_40_70);
    _mm256_storeu_pd(C+4+ldc, c_41_71);
    _mm256_storeu_pd(C+4+ldc*2, c_42_72);
}



// This is the routine to handle edge cases, needs to implement
// Maybe we can pack A and B here to increase performance
void dgemm_edge(int lda, int m, int n, int k, double *A, double *B, double* C)
{
    if (m == 0 || n == 0) return;
    // This is for debug purpose:
    if (DEBUG) printf("Updaing edge case m = %d, n = %d, k = %d.\n", m, n, k);
    // This routine will do a little optimization by updating 2X2 blocks
    int r, c, p;
    __m128d c_00_10, c_01_11,
            a_0p_1p,
            b_p0, b_p1;
    for(r = 0; r < m/2*2; r+=2)
    {
        for(c = 0; c < n/2*2; c+=2)
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
        for(c=0; c<n/2*2; c+=2)
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

        if (n % 2)
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
 */
void do_block(int lda, int m, int n, int k, double *A, double *B, double *C, int need_to_packB)
{
    int i, j;
    double packedA[m*k], packedB[block_small*nb];

    for (j=0; j<n/4*4; j+=4)
    {        /* Loop over the columns of C, unrolled by 4 */
        if (need_to_packB) copyB(k, B+j*lda, lda, &packedB[j*k]);
        for (i=0; i<m/8*8; i+=8)
        {        /* Loop over the rows of C, unrolled by 8 */
            if (j == 0) copyA8(k, A+i, lda, &packedA[i*k]);
            update8X4(k, &packedA[i*k], 8, &packedB[j*k], k, C+i+j*lda, lda);
        }
    }


    if (m%8!=0 && n%4 == 0)
    {
        // Need to update a bottom row block
        int row_index = m/8*8;
        dgemm_edge(lda, m - row_index, n, k, A+row_index, B, C+row_index);
    }
    else if (m%8!=0 && n%4 != 0)
    {
        int row_index = m/8*8, col_index = n/4*4;
        if (n%4 == 3)
        {
            double packedB3[block_small*3];
            copyB3(k, B+col_index*lda, lda, &packedB3[0]);
            for (i=0; i<m/8*8; i+=8)
            {        /* Loop over the rows of C, unrolled by 8 */
                if (j == 0) copyA8(k, A+i, lda, &packedA[i*k]);
                update8X3(k, &packedA[i*k], 8, &packedB3[0], k, C+i+j*lda, lda);
            }
        }else dgemm_edge(lda, row_index, n-col_index, k, A, B+lda*col_index, C+lda*col_index);
        // Need to update a bottom row block and a tailing column block
        dgemm_edge(lda, m-row_index, n, k, A+row_index, B, C+row_index);

    }
        // Here we rule out the case that m%8 == 0 and n%4 != 0 cause that's not possible
    else if (m%8 == 0 && n%4!=0)
    {
        int col_index = n/4*4;

        dgemm_edge(lda, m, n-col_index, k, A, B+lda*col_index, C+lda*col_index);
    }

}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 * Two levels of blocking- only slight improvement
 * */
void square_dgemm_blocked(int lda, double *A, double *B, double *C)
{
    int i, j, pb, ib;

    for (j=0; j<lda; j+=block_small)
    {
        // Column blocks of A
        pb = min( lda-j, block_small );
        for (i=0; i<lda; i+=block_small)
        {
            // Row blocks of C
            ib = min(lda-i, block_small);
            do_block(lda, ib, lda, pb, A+i+j*lda, B+j, C+i, i==0);
        }
    }
}
