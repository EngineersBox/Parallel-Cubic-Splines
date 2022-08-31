package com.engineersbox.pcsplines;

import java.awt.geom.Path2D;
import java.awt.geom.Point2D;

public class Point2DPath extends Path2D.Double {

    public Point2DPath() {
        super();
    }

    /**
     * Adds a point to the path by moving to the specified coordinates specified in double precision.
     *
     * @param point Path point
     */
    public final synchronized void moveTo(final Point2D point) {
        super.moveTo(point.getX(), point.getY());
    }

    /**
     * Adds a point to the path by drawing a straight line from the current coordinates
     * to the new specified coordinates specified in double precision.
     *
     * @param point Path point
     */
    public final synchronized void lineTo(final Point2D point) {
        super.lineTo(point.getX(), point.getY());
    }

    /**
     * Adds a curved segment, defined by two new points, to the path by drawing a
     * Quadratic curve that intersects both the current coordinates and the
     * specified coordinates (x2,y2), using the specified point (x1,y1) as a
     * quadratic parametric control point. All coordinates are specified in double
     * precision.
     *
     * @param point1 Path point 1
     * @param point2 Path point 2
     */
    public final synchronized void quadTo(final Point2D point1,
                                          final Point2D point2) {
        super.quadTo(
                point1.getX(), point1.getY(),
                point2.getX(), point2.getY()
        );
    }

    /**
     * Adds a curved segment, defined by three new points, to the path by drawing a
     * Bézier curve that intersects both the current coordinates and the specified
     * coordinates (x3,y3), using the specified points (x1,y1) and (x2,y2) as
     * Bézier control points. All coordinates are specified in double precision.
     *
     * @param control1 Control point 1
     * @param control2 Control point 2
     * @param point Path point
     */
    public final synchronized void curveTo(final Point2D control1,
                                           final Point2D control2,
                                           final Point2D point) {
        super.curveTo(
                control1.getX(), control1.getY(),
                control2.getX(), control2.getY(),
                point.getX(), point.getY()
        );
    }
}