package com.vv.tutorial.opencvyolo;

import org.opencv.core.Scalar;

public class Utils {


    // Generate unique distinct color based on Class Id
    public static Scalar generateColor(int classId) {
        float hue = (classId * 360f / 80f); // 0–360
        float saturation = 0.9f;
        float value = 0.9f;

        // Convert HSV to RGB
        float c = value * saturation;
        float x = c * (1 - Math.abs((hue / 60f) % 2 - 1));
        float m = value - c;

        float r, g, b;

        if (hue < 60) {
            r = c;
            g = x;
            b = 0;
        } else if (hue < 120) {
            r = x;
            g = c;
            b = 0;
        } else if (hue < 180) {
            r = 0;
            g = c;
            b = x;
        } else if (hue < 240) {
            r = 0;
            g = x;
            b = c;
        } else if (hue < 300) {
            r = x;
            g = 0;
            b = c;
        } else {
            r = c;
            g = 0;
            b = x;
        }

        return new Scalar(
                (b + m) * 255.0,
                (g + m) * 255.0,
                (r + m) * 255.0
        );
    }

}
