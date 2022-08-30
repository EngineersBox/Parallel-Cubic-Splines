package com.engineersbox.pcsplines;

import com.engineersbox.pcsplines.utils.OpenCLUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Path2D;
import java.awt.geom.Point2D;
import java.util.ArrayList;

import static org.jocl.CL.*;

/**
 * Interpolates given points by a bezier curve. The first
 * and the last two points are interpolated by a quadratic
 * bezier curve; the other points by a cubic bezier curve.
 *
 * Let p a list of given points and b the calculated bezier points,
 * then one get the whole curve by:
 *
 * sharedPath.moveTo(p[0])
 * sharedPath.quadTo(b[0].x, b[0].getY(), p[1].x, p[1].getY());
 *
 * for(int i = 2; i < p.length - 1; i++ ) {
 *    Point b0 = b[2*i-3];
 *    Point b1 = b[2*i-2];
 *    sharedPath.curveTo(b0.x, b0.getY(), b1.x, b1.getY(), p[i].x, p[i].getY());
 * }
 *
 * sharedPath.quadTo(b[b.length-1].x, b[b.length-1].getY(), p[n - 1].x, p[n - 1].getY());
 *
 * @author krueger
 */
public class BezierCurve extends JComponent {

    private static final String COMPUTE_CONTROL_POINTS_KERNEL_NAME = "computeBezierControlPoints";
    private static final float AP = 0.5f;
    private final Point2D[] initialPoints;
    private final OpenCLParams openclParams;
    private Point2D[] bezierPoints;

    public BezierCurve(final Point2D[] points,
                       final OpenCLParams openclParams) {
        this.initialPoints = points;
        this.openclParams = openclParams;
        this.bezierPoints = calculateControlPointsGPU();
        for (final Point2D point : this.bezierPoints) {
            System.out.printf("(%f, %f)%n", point.getX(), point.getY());
        }
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
                Pair.of(Pointer.to(deviceControlPoints), Sizeof.cl_mem)
        );
        if (result != CL_SUCCESS) {
            clReleaseMemObject(deviceInitialPoints);
            clReleaseMemObject(deviceControlPoints);
            throw new IllegalStateException("Unable to bind kernel args: " + stringFor_errorCode(result));
        }
        result = clEnqueueNDRangeKernel(
                this.openclParams.getQueue(),
                kernel,
                1,
                null,
                new long[]{this.initialPoints.length - 2},
                new long[]{1},
                0,
                null,
                null
        );
        if (result != CL_SUCCESS) {
            clReleaseMemObject(deviceInitialPoints);
            clReleaseMemObject(deviceControlPoints);
            throw new IllegalStateException("Unable to execute kernel: " + stringFor_errorCode(result));
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
        if (result != CL_SUCCESS) {
            clReleaseMemObject(deviceInitialPoints);
            clReleaseMemObject(deviceControlPoints);
            throw new IllegalStateException("Unable to read result from kernel: " + stringFor_errorCode(result));
        }
        final Point2D[] finalControlPoints = new Point2D[2 * (this.initialPoints.length - 2)];
        for (int i = 0; i < 2 * (this.initialPoints.length - 2); i += 2) {
            finalControlPoints[i] = new Point2D.Double(
                    controlPoints[(i * 2)],
                    controlPoints[(i * 2) + 1]
            );
        }
        return finalControlPoints;
    }

    /**
     * Creates a new Bezier curve.
     * @param points
     */
//    public BezierCurve(final Point2D[] points) {
////        points = ArrayUtils.addAll(new Point2D[]{
////                points[0],
////                points[0]
////        }, points);
//        this.initialPoints = points;
//        final int n = points.length;
//        if (n < 3) {
//            // Cannot create bezier with less than 3 points
//            return;
//        }
//        this.bezierPoints = new Point[2 * (n - 2)];
//        final double[][] pointAttributes = new double[n - 2][6];
//        double paX, paY;
//        double pbX = points[0].getX();
//        double pbY = points[0].getY();
//        double pcX = points[1].getX();
//        double pcY = points[1].getY();
//        for (int i = 0; i < n - 2; i++) {
//            paX = pbX;
//            paY = pbY;
//            pbX = pcX;
//            pbY = pcY;
//            pcX = points[i + 2].getX();
//            pcY = points[i + 2].getY();
//            pointAttributes[i] = new double[]{paX, paY, pbX, pbY, pcX, pcY};
//            final double abX = pbX - paX;
//            final double abY = pbY - paY;
//            double acX = pcX - paX;
//            double acY = pcY - paY;
//            final double lac = Math.sqrt(acX * acX + acY * acY);
//            acX = acX / lac;
//            acY = acY / lac;
//
//            double proj = abX * acX + abY * acY;
//            proj = proj < 0 ? -proj : proj;
//            double apX = proj * acX;
//            double apY = proj * acY;
//
//            final double p1X = pbX - BezierCurve.AP * apX;
//            final double p1Y = pbY - BezierCurve.AP * apY;
//            this.bezierPoints[2 * i] = new Point((int) p1X, (int) p1Y);
//
//            acX = -acX;
//            acY = -acY;
//            final double cbX = pbX - pcX;
//            final double cbY = pbY - pcY;
//            proj = cbX * acX + cbY * acY;
//            proj = proj < 0 ? -proj : proj;
//            apX = proj * acX;
//            apY = proj * acY;
//
//            final double p2X = pbX - BezierCurve.AP * apX;
//            final double p2Y = pbY - BezierCurve.AP * apY;
//            this.bezierPoints[2 * i + 1] = new Point((int) p2X, (int) p2Y);
//        }
//        for (int i = 0; i < pointAttributes.length; i++) {
//            System.out.printf(
//                    "[%d] paX: %f paY: %f pbX: %f pbY: %f pcX: %f pcY: %f%n",
//                    i,
//                    pointAttributes[i][0],
//                    pointAttributes[i][1],
//                    pointAttributes[i][2],
//                    pointAttributes[i][3],
//                    pointAttributes[i][4],
//                    pointAttributes[i][5]
//            );
//        }
//    }

    /**
     * Returns the calculated bezier points.
     * @return the calculated bezier points
     */
    public Point2D[] getPoints() {
        return this.bezierPoints;
    }

    /**
     * Returns the number of bezier points.
     * @return number of bezier points
     */
    public int getPointCount() {
        return this.bezierPoints.length;
    }

    /**
     * Returns the bezier points at position i.
     * @param i
     * @return the bezier point at position i
     */
    public Point2D getPoint(final int i) {
        return this.bezierPoints[i];
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
                    }, new OpenCLParams("/kernels/bezier_points.ocl")
            );
            contentPane.add(bezierCurve);
            this.jframe.setMinimumSize(new Dimension(100, 100));
            this.jframe.setVisible(true);
            this.jframe.pack();

        }
    }

}
