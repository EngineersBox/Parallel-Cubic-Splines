package com.engineersbox.pcsplines;

import com.engineersbox.pcsplines.utils.OpenCLUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

import java.util.List;

import static org.jocl.CL.*;

public class GaussJordanElimination {

    private static final int BLOCK_SIZE = 16;
    private static final String SCALE_ROW_KERNEL_NAME = "scaleRow";
    private static final String SUBTRACT_ROW_KERNEL_NAME = "subtractRow";

    private final int rows;
    private final int cols;

    private final OpenCLParams openclParams;
    private final cl_mem M;
    private final cl_mem P;
    private final int matrixBytes;

    public GaussJordanElimination(final double[][] matrix,
                                  final int rows,
                                  final int cols,
                                  final OpenCLParams openclParams) {
        this.rows = rows;
        this.cols = cols;
        this.openclParams = openclParams;

        this.matrixBytes = rows * cols * Sizeof.cl_double;
        this.M = clCreateBuffer(
                this.openclParams.getContext(),
                CL_MEM_READ_WRITE,
                this.matrixBytes,
                null,
                null
        );
        this.P = clCreateBuffer(
                this.openclParams.getContext(),
                CL_MEM_READ_WRITE,
                this.matrixBytes,
                null,
                null
        );
        clCheck(OpenCLUtils.writeBuffer2D(
                this.openclParams.getQueue(),
                this.M,
                matrix
        ));
        clCheck(OpenCLUtils.writeBuffer2D(
                this.openclParams.getQueue(),
                this.P,
                matrix
        ));
    }

    private void clCheck(final int result) {
        if (result != CL_SUCCESS) {
            clReleaseMemObject(this.M);
            clReleaseMemObject(this.P);
            throw new RuntimeException("Could not run kernel");
        }
    }

    private void scaleRow(final int currentCol) {
        final cl_kernel kernel = this.openclParams.getKernel(SCALE_ROW_KERNEL_NAME);
        OpenCLUtils.bindKernelArgs(
                kernel,
                List.of(
                        Pair.of(Pointer.to(this.M), Sizeof.cl_mem),
                        Pair.of(Pointer.to(new int[]{this.rows}), Sizeof.cl_uint),
                        Pair.of(Pointer.to(new int[]{this.cols}), Sizeof.cl_uint),
                        Pair.of(Pointer.to(this.P), Sizeof.cl_mem),
                        Pair.of(Pointer.to(new int[]{currentCol}), Sizeof.cl_int)
                )
        );
        clCheck(clEnqueueNDRangeKernel(
                this.openclParams.getQueue(),
                kernel,
                2,
                null,
                new long[]{
                        BLOCK_SIZE,
                        BLOCK_SIZE
                },
                new long[]{
                        (long) Math.ceil((float) this.cols / BLOCK_SIZE),
                        (long) Math.ceil((float) this.rows / BLOCK_SIZE)
                },
                0,
                null, null
        ));
        clCheck(clFinish(this.openclParams.getQueue()));
        clCheck(clEnqueueCopyBuffer(
                this.openclParams.getQueue(),
                this.P, this.M,
                0, 0,
                this.matrixBytes,
                0,
                null, null
        ));
    }

    private void subtractRow(final int currentCol) {
        final cl_kernel kernel = this.openclParams.getKernel(SUBTRACT_ROW_KERNEL_NAME);
        OpenCLUtils.bindKernelArgs(
                kernel,
                List.of(
                        Pair.of(Pointer.to(this.M), Sizeof.cl_mem),
                        Pair.of(Pointer.to(new int[]{this.rows}), Sizeof.cl_uint),
                        Pair.of(Pointer.to(new int[]{this.cols}), Sizeof.cl_uint),
                        Pair.of(Pointer.to(this.P), Sizeof.cl_mem),
                        Pair.of(Pointer.to(new int[]{currentCol}), Sizeof.cl_int)
                )
        );
        clCheck(clEnqueueNDRangeKernel(
                this.openclParams.getQueue(),
                kernel,
                2,
                null,
                new long[]{
                        BLOCK_SIZE,
                        BLOCK_SIZE
                },
                new long[]{
                        (long) Math.ceil((float) this.cols / BLOCK_SIZE),
                        (long) Math.ceil((float) this.rows / BLOCK_SIZE)
                },
                0,
                null, null
        ));
        clCheck(clFinish(this.openclParams.getQueue()));
        clCheck(clEnqueueCopyBuffer(
                this.openclParams.getQueue(),
                this.P, this.M,
                0, 0,
                this.matrixBytes,
                0,
                null, null
        ));
    }

    public double[][] calculate() {
        // TODO: Implement rest of GJE
        for (int currentCol = 0; currentCol < this.rows; currentCol++) {
            scaleRow(currentCol);
            subtractRow(currentCol);
        }
        final double[][] result = new double[this.cols][this.rows];
        clCheck(OpenCLUtils.readBuffer2D(
                this.openclParams.getQueue(),
                this.M,
                result
        ));
        return result;
    }

    public static String printMatrix(final double[][] matrix,
                                   final int rows,
                                   final int cols) {
        final StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sb.append(matrix[i][j]);
                sb.append(" ");
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        final OpenCLParams params = new OpenCLParams("E:\\COMP4610\\assign1\\src\\main\\resources\\kernels\\gauss_jordan_elimination.cl");
        final double[][] input = new double[][]{
                {1,2,3},
                {4,5,6},
                {7,8,9}
        };
        System.out.println("Input: \n" + printMatrix(input, 3, 3));
        final GaussJordanElimination gje = new GaussJordanElimination(
                input,
                3, 3,
                params
        );
        final double[][] result = gje.calculate();
        System.out.println("Output: \n" + printMatrix(result, 3, 3));
    }
}
