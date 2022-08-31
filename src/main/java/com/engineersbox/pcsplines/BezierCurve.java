package com.engineersbox.pcsplines;

import com.engineersbox.pcsplines.utils.OpenCLUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Path2D;
import java.awt.geom.Point2D;

import static org.jocl.CL.*;

public class BezierCurve extends JComponent {

    private static final String COMPUTE_CONTROL_POINTS_KERNEL_NAME = "computeBezierControlPoints";
    private static final float AP = 0.5f;
    private final float smoothFactor;
    private final Point2D[] initialPoints;
    private final OpenCLParams openclParams;
    private Point2D[] bezierPoints;

    public BezierCurve(final Point2D[] points,
                       final float smoothFactor,
                       final OpenCLParams openclParams) {
        this.initialPoints = points;
        this.smoothFactor = smoothFactor;
        this.openclParams = openclParams;
        this.bezierPoints = calculateControlPointsGPU();
    }

    private Point2D[] calculateControlPointsGPU() {
        final int flattenedPointsLength = this.initialPoints.length * 2;
        final double[] flattenedPoints = new double[flattenedPointsLength];
        final int controlPointsLength = 4 * (this.initialPoints.length - 2);
        final double[] controlPoints = new double[controlPointsLength];
        for (int i = 0; i < this.initialPoints.length; i++) {
            flattenedPoints[(i * 2)] = this.initialPoints[i].getX();
            flattenedPoints[(i * 2) + 1] = this.initialPoints[i].getY();
        }
        final cl_mem deviceInitialPoints = clCreateBuffer(
                this.openclParams.getContext(),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_double * (long) flattenedPointsLength,
                Pointer.to(flattenedPoints),
                null
        );
        int result = clEnqueueWriteBuffer(
                this.openclParams.getQueue(),
                deviceInitialPoints,
                CL_TRUE,
                0,
                Sizeof.cl_double * (long) flattenedPointsLength,
                Pointer.to(flattenedPoints),
                0,
                null,
                null
        );
        if (result != CL_SUCCESS) {
            clReleaseMemObject(deviceInitialPoints);
            this.openclParams.releaseAll(null);
            throw new IllegalStateException("Unable to write flattened points to device: " + stringFor_errorCode(result));
        }
        final cl_mem deviceControlPoints = clCreateBuffer(
                this.openclParams.getContext(),
                CL_MEM_READ_WRITE,
                Sizeof.cl_double * (long) controlPointsLength,
                null,
                null
        );
        final cl_kernel kernel = this.openclParams.getKernel(COMPUTE_CONTROL_POINTS_KERNEL_NAME);
        result = OpenCLUtils.bindKernelArgs(
                kernel,
                Pair.of(Pointer.to(deviceInitialPoints), Sizeof.cl_mem),
                Pair.of(Pointer.to(deviceControlPoints), Sizeof.cl_mem),
                Pair.of(Pointer.to(new float[]{this.smoothFactor}), Sizeof.cl_float)
        );
        if (result != CL_SUCCESS) {
            clReleaseMemObject(deviceInitialPoints);
            clReleaseMemObject(deviceControlPoints);
            this.openclParams.releaseAll(kernel);
            throw new IllegalStateException("Unable to bind kernel args: " + stringFor_errorCode(result));
        }
        final long maxWorkGroupSize = this.openclParams.getMaxWorkGroupSize(kernel);
        System.out.println("Max work group size: " + maxWorkGroupSize);
        result = clEnqueueNDRangeKernel(
                this.openclParams.getQueue(),
                kernel,
                1,
                null,
                new long[]{Math.min(
                        2 * (this.initialPoints.length - 2L),
                        maxWorkGroupSize
                )},
                new long[]{1},
                0,
                null,
                null
        );
        if (result != CL_SUCCESS) {
            clReleaseMemObject(deviceInitialPoints);
            clReleaseMemObject(deviceControlPoints);
            this.openclParams.releaseAll(kernel);
            throw new IllegalStateException("Unable to execute kernel: " + stringFor_errorCode(result));
        }
        result = clFinish(this.openclParams.getQueue());
        if (result != CL_SUCCESS) {
            clReleaseMemObject(deviceInitialPoints);
            clReleaseMemObject(deviceControlPoints);
            this.openclParams.releaseAll(kernel);
            throw new IllegalStateException("Kernel did not finish successfully: " + stringFor_errorCode(result));
        }
        result = clEnqueueReadBuffer(
                this.openclParams.getQueue(),
                deviceControlPoints,
                CL_TRUE,
                0,
                Sizeof.cl_double * (long) controlPointsLength,
                Pointer.to(controlPoints),
                0,
                null,
                null
        );
        clReleaseMemObject(deviceInitialPoints);
        clReleaseMemObject(deviceControlPoints);
        this.openclParams.releaseAll(kernel);
        if (result != CL_SUCCESS) {
            throw new IllegalStateException("Unable to read result from kernel: " + stringFor_errorCode(result));
        }
        final Point2D[] finalControlPoints = new Point2D[2 * (this.initialPoints.length - 2)];
        for (int i = 0; i < 2 * (this.initialPoints.length - 2); i++) {
            finalControlPoints[i] = new Point2D.Double(
                    controlPoints[(i * 2)],
                    controlPoints[(i * 2) + 1]
            );
        }
        return finalControlPoints;
    }

    public void draw(final Graphics2D g) {
        final Path2D path = new Path2D.Double();
        path.moveTo(
                this.initialPoints[0].getX(),
                this.initialPoints[0].getY()
        );
        path.quadTo(
                this.bezierPoints[0].getX(),
                this.bezierPoints[0].getY(),
                this.bezierPoints[1].getX(),
                this.bezierPoints[1].getY()
        );

        for(int i = 2; i < this.initialPoints.length - 1; i++ ) {
            final Point2D b0 = this.bezierPoints[2*i-3];
            final Point2D b1 = this.bezierPoints[2*i-2];
            path.curveTo(
                    b0.getX(),
                    b0.getY(),
                    b1.getX(),
                    b1.getY(),
                    this.initialPoints[i].getX(),
                    this.initialPoints[i].getY()
            );
        }

        path.quadTo(
                this.bezierPoints[this.bezierPoints.length - 1].getX(),
                this.bezierPoints[this.bezierPoints.length - 1].getY(),
                this.bezierPoints[this.initialPoints.length - 1].getX(),
                this.bezierPoints[this.initialPoints.length - 1].getY()
        );
        g.draw(path);
    }

    @Override
    public void paintComponent(final Graphics g) {
        final Graphics2D g2d = (Graphics2D) g;
        draw(g2d);
    }

    public static class TestMain implements Runnable {

        private JFrame jframe;

        public TestMain() {
            SwingUtilities.invokeLater(this);
        }

        public static void main(String[] args) {
            new TestMain();
        }

        @Override
        public void run() {
            this.jframe = new JFrame("Test Main");
            this.jframe.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            this.jframe.setPreferredSize(new Dimension(800, 600));
            final Container contentPane = this.jframe.getContentPane();
            final BezierCurve bezierCurve = new BezierCurve(
                    new Point2D[]{
                            new Point2D.Double(50, 53),
                            new Point2D.Double(120, 150),
                            new Point2D.Double(200, 70),
                            new Point2D.Double(250, 120),
                            new Point2D.Double(290, 102),
                            new Point2D.Double(310, 52),
                            new Point2D.Double(370, 214),
                    },
                    0.5f,
                    new OpenCLParams("/kernels/bezier_points.ocl")
            );
            contentPane.add(bezierCurve);
            this.jframe.setMinimumSize(new Dimension(100, 100));
            this.jframe.setVisible(true);
            this.jframe.pack();

        }
    }

}
