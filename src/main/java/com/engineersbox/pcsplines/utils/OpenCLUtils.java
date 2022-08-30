package com.engineersbox.pcsplines.utils;

import org.apache.commons.lang3.tuple.Pair;
import org.jocl.*;

import java.util.List;

import static org.jocl.CL.*;

public class OpenCLUtils {

    private OpenCLUtils() {
        throw new IllegalStateException("Utility class");

    }

    public static int writeBuffer2D(final cl_command_queue commandQueue,
                                    final cl_mem buffer,
                                    final double[][] array) {
        long byteOffset = 0;
        for (final double[] element : array) {
            final int bytes = element.length * Sizeof.cl_double;
            final int result = clEnqueueWriteBuffer(
                    commandQueue,
                    buffer,
                    CL_TRUE,
                    byteOffset,
                    bytes,
                    Pointer.to(element),
                    0,
                    null,
                    null
            );
            if (result != CL_SUCCESS) {
                return result;
            }
            byteOffset += bytes;
        }
        return CL_SUCCESS;
    }

    public static int readBuffer2D(final cl_command_queue commandQueue,
                                   final cl_mem buffer,
                                   final double[][] array)  {
        long byteOffset = 0;
        for (final double[] element : array) {
            final int bytes = element.length * Sizeof.cl_double;
            final int result = clEnqueueReadBuffer(
                    commandQueue,
                    buffer,
                    CL_TRUE,
                    byteOffset,
                    bytes,
                    Pointer.to(element),
                    0,
                    null,
                    null
            );
            if (result != CL_SUCCESS) {
                return result;
            }
            byteOffset += bytes;
        }
        return CL_SUCCESS;
    }

    @SafeVarargs
    public static int bindKernelArgs(final cl_kernel kernel,
                                     final Pair<Pointer, Integer> ...args) {
        int result;
        for (int i = 0; i < args.length; i++) {
            final Pair<Pointer, Integer> arg = args[i];
            result = clSetKernelArg(kernel, i, arg.getValue(), arg.getKey());
            if (result != CL_SUCCESS) {
                return result;
            }
        }
        return CL_SUCCESS;
    }

}
