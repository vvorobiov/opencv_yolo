package com.vv.tutorial.opencvyolo;

import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static com.vv.tutorial.opencvyolo.Constants.COCO_CLASS_NAMES;
import static com.vv.tutorial.opencvyolo.Constants.COLORS;

/**
 * YOLO Object Detector for YOLOv11 models.
 * Supports ONNX format with proper handling of [1, 84, 8400] output tensors.
 */
public class YoloObjectDetector {

    /**
     * --------- CONFIGURATION ---------
     */
    // Full path to a yolo11*.onnx model in your file system
    // Full path to *.mp4 video file in your file system
    static final String MODEL_PATH = "C:\\Users\\vadym\\dev\\models\\yolo11n.onnx";
    static final String VIDEO_PATH = "C:\\Users\\vadym\\dev\\videos\\v9.mp4";

    /**
     * --------- DETECTION PARAMS ------
     */
    static final float CONF_THRESHOLD = 0.25f;
    static final float NMS_THRESHOLD = 0.45f;
    static final Size INPUT_SIZE = new Size(640, 640);
    static final boolean USE_LETTERBOX = true;
    static final boolean SWAP_RB = true;      // YOLO models are trained on RGB
    static final double SCALE = 1.0 / 255.0;

    /**
     * Main inference pipeline.
     */
    static List<Detection> infer(Net net, Mat bgr, int numClasses) {

        // Preprocess with letterbox or simple resize
        // Needed for proper resizing the frame if not 640*640
        Preproc p = USE_LETTERBOX ? letterbox(bgr, INPUT_SIZE) : resizeNoPad(bgr, INPUT_SIZE);

        // Create blob from image
        Mat blob = Dnn.blobFromImage(
                p.image, SCALE, INPUT_SIZE, new Scalar(0, 0, 0), SWAP_RB, false);

        // Run inference
        net.setInput(blob);
        List<Mat> outputs = new ArrayList<>();
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Parse YOLO output (handles [1, 84, 8400] format)
        Mat out = outputs.get(0);
        Mat out2D = squeezeTo2D(out);

        // Parse detections
        List<Detection> all = parseYoloRows(out2D, numClasses, CONF_THRESHOLD);

        // Transform boxes back to original image coordinates
        for (Detection d : all) {
            Rect2d box = d.box;
            Point p1 = unletterboxPoint(new Point(box.x, box.y), bgr.size(), p);
            Point p2 = unletterboxPoint(new Point(box.x + box.width, box.y + box.height), bgr.size(), p);
            d.box = new Rect2d(
                    Math.max(0, p1.x),
                    Math.max(0, p1.y),
                    Math.min(bgr.cols() - p1.x, p2.x - p1.x),
                    Math.min(bgr.rows() - p1.y, p2.y - p1.y)
            );
        }

        // Apply Non-Maximum Suppression
        List<Detection> detections = nms(all, NMS_THRESHOLD);

        // Clean up
        blob.release();
        out.release();
        out2D.release();

        return detections;
    }

    /**
     * Non-Maximum Suppression to remove overlapping detections.
     */
    static List<Detection> nms(List<Detection> dets, float nmsTh) {
        if (dets.isEmpty()) return dets;

        List<Rect2d> boxes = dets.stream().map(d -> d.box).collect(Collectors.toList());
        MatOfRect2d matBoxes = new MatOfRect2d();
        matBoxes.fromList(boxes);

        float[] scoresArr = new float[dets.size()];
        for (int i = 0; i < dets.size(); i++) scoresArr[i] = dets.get(i).confidence;
        MatOfFloat scores = new MatOfFloat(scoresArr);

        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(matBoxes, scores, CONF_THRESHOLD, nmsTh, indices);

        int[] keep = indices.toArray();
        List<Detection> out = new ArrayList<>(keep.length);
        for (int idx : keep) out.add(dets.get(idx));

        return out;
    }

    /**
     * Parse YOLO output tensor rows.
     * Handles YOLOv8/v11 format: [cx, cy, w, h, class_scores...]
     * And YOLOv5 format: [cx, cy, w, h, objectness, class_scores...]
     */
    static List<Detection> parseYoloRows(Mat out2D, int numClasses, float confTh) {
        Mat m = new Mat();
        out2D.convertTo(m, CvType.CV_32FC1);

        int rows = m.rows();
        int cols = m.cols();
        float[] data = new float[rows * cols];
        m.get(0, 0, data);

        // Detect format: YOLOv5 has objectness, YOLOv8/v11 doesn't
        boolean hasObjectness = (cols == numClasses + 5);
        int classStartIdx = hasObjectness ? 5 : 4;

        List<Detection> dets = new ArrayList<>();

        for (int i = 0; i < rows; i++) {
            int base = i * cols;

            float cx = data[base + 0];
            float cy = data[base + 1];
            float w = data[base + 2];
            float h = data[base + 3];

            // Get objectness score if present
            float objScore = hasObjectness ? data[base + 4] : 1.0f;

            // Find best class
            int bestClass = -1;
            float bestScore = 0f;
            for (int c = 0; c < numClasses; c++) {
                float score = data[base + classStartIdx + c];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = c;
                }
            }

            // Calculate final confidence
            float conf = hasObjectness ? (objScore * bestScore) : bestScore;

            if (conf < confTh) continue;

            // Convert from center format to corner format
            double x1 = cx - w / 2.0;
            double y1 = cy - h / 2.0;
            dets.add(new Detection(new Rect2d(x1, y1, w, h), conf, bestClass, getClassName(bestClass)));
        }

        m.release();
        return dets;
    }

    static String getClassName(int classId) {
        if (classId >= 0 && classId < COCO_CLASS_NAMES.size()) {
            return COCO_CLASS_NAMES.get(classId);
        }
        return "Class " + classId;
    }

    /**
     * Detection result containing bounding box, confidence, and class info.
     */
    static class Detection {
        private Rect2d box;
        private float confidence;
        private int classId;
        private String className;

        public Detection(Rect2d box, float confidence, int classId, String className) {
            this.box = box;
            this.confidence = confidence;
            this.classId = classId;
            this.className = className;
        }

        public Rect2d getBox() {
            return box;
        }

        public float getConfidence() {
            return confidence;
        }

        public int getClassId() {
            return classId;
        }

        public String getClassName() {
            return className;
        }

        @Override
        public String toString() {
            return String.format("%s (%.2f): [%.0f, %.0f, %.0f, %.0f]",
                    className, confidence, box.x, box.y, box.width, box.height);
        }
    }

    /**
     * Preprocessing result with transformation parameters.
     */
    static class Preproc {
        Mat image;
        double r;
        int top, bottom, left, right;
        Size outSize;

        Preproc(Mat img, double r, int t, int b, int l, int rgt, Size out) {
            this.image = img;
            this.r = r;
            this.top = t;
            this.bottom = b;
            this.left = l;
            this.right = rgt;
            this.outSize = out;
        }
    }

    static Preproc resizeNoPad(Mat img, Size out) {
        Mat resized = new Mat();
        Imgproc.resize(img, resized, out, 0, 0, Imgproc.INTER_LINEAR);
        return new Preproc(resized, 1.0, 0, 0, 0, 0, out);
    }

    /**
     * Letterbox preprocessing - maintains aspect ratio with padding.
     * This is the recommended approach for YOLO models.
     */
    static Preproc letterbox(Mat sourceImg, Size targetSize) {
        int sourceHeight = sourceImg.rows();
        int sourceWidth = sourceImg.cols();
        double r = Math.min(targetSize.width / sourceWidth, targetSize.height / sourceHeight);
        int adjustedWidth = (int) Math.round(sourceWidth * r);
        int adjustedHeight = (int) Math.round(sourceHeight * r);

        Mat resizedImg = new Mat();
        Imgproc.resize(sourceImg, resizedImg, new Size(adjustedWidth, adjustedHeight), 0, 0, Imgproc.INTER_LINEAR);

        int deltaWidth = (int) (targetSize.width - adjustedWidth);
        int deltaHeight = (int) (targetSize.height - adjustedHeight);

        int top = deltaHeight / 2;
        int bottom = deltaHeight - top;
        int left = deltaWidth / 2;
        int right = deltaWidth - left;

        Mat borderedImg = new Mat();
        Core.copyMakeBorder(resizedImg, borderedImg, top, bottom, left, right,
                Core.BORDER_CONSTANT, new Scalar(114, 114, 114));

        resizedImg.release();
        return new Preproc(borderedImg, r, top, bottom, left, right, targetSize);
    }

    /**
     * Transform point from preprocessed image space back to original image space.
     */
    static Point unletterboxPoint(Point p, Size orig, Preproc pp) {
        double x = (p.x - pp.left) / pp.r;
        double y = (p.y - pp.top) / pp.r;
        return new Point(x, y);
    }

    /**
     * Convert YOLO output tensor to 2D matrix.
     * Handles format [1, 84, 8400] -> [8400, 84]
     */
    static Mat squeezeTo2D(Mat yoloOutput) {
        int dims = yoloOutput.dims();
        if (dims == 2) return yoloOutput;

        if (dims == 3) {
            int d0 = yoloOutput.size(0);
            int d1 = yoloOutput.size(1);
            int d2 = yoloOutput.size(2);

            // Handle [1, features, detections] -> [detections, features]
            if (d0 == 1 && d1 >= 5 && d2 > 0) {
                int[] newSize = {d1, d2};
                Mat reshaped = yoloOutput.reshape(1, newSize);
                Mat transposed = new Mat();
                Core.transpose(reshaped, transposed);
                reshaped.release();
                return transposed;
            }
        }

        // Fallback: try to find valid column count
        int total = (int) yoloOutput.total();
        for (int cols = 6; cols <= 2000; cols++) {
            if (total % cols == 0) {
                return yoloOutput.reshape(1, total / cols);
            }
        }
        return yoloOutput.reshape(1, yoloOutput.rows());
    }

    /**
     * Draw detection boxes and labels on image.
     */
    static void drawDetections(Mat img, List<Detection> dets) {
        for (Detection d : dets) {
            Rect r = new Rect(
                    (int) Math.max(0, Math.round(d.box.x)),
                    (int) Math.max(0, Math.round(d.box.y)),
                    (int) Math.max(0, Math.round(d.box.width)),
                    (int) Math.max(0, Math.round(d.box.height))
            );

            // Draw bounding box
            Imgproc.rectangle(img, r, COLORS.get(d.classId), 2);

            // Draw label background and text
            String label = d.className + String.format(" %.2f", d.confidence);
            int[] baseLine = new int[1];
            Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);
            int top = Math.max(r.y, (int) labelSize.height);

            Imgproc.rectangle(img,
                    new Point(r.x, top - labelSize.height),
                    new Point(r.x + labelSize.width, top + baseLine[0]),
                    COLORS.get(d.classId), Imgproc.FILLED);
            Imgproc.putText(img, label, new Point(r.x, top),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 0), 1);
        }
    }

    // --------- MAIN ---------
    public static void main(String[] args) {

        // Load opencv*.dll
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load YOLO model
        Net net = Dnn.readNetFromONNX(MODEL_PATH);
        net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        net.setPreferableTarget(Dnn.DNN_TARGET_CPU);

        // Open video
        VideoCapture cap = new VideoCapture(VIDEO_PATH);
        if (!cap.isOpened()) {
            System.err.println("Error opening video file");
            return;
        }

        int frameWidth = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
        int frameHeight = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(Videoio.CAP_PROP_FPS);
        System.out.printf("Video: %dx%d @ %.2f FPS%n", frameWidth, frameHeight, fps);

        Mat frame = new Mat();
        Mat roi = new Mat();
        int frameCount = 0;

        // Only needed if you want to specify exact area of interest a frame.
        // Exact rectangle size has to be determined for each particular video
        // Use together with frame.submat() (uncomment below)
        //Rect roiRect = new Rect(0, 450, 540, 450);

        // Process frames
        while (cap.read(frame)) {
            if (frame.empty()) break;

            roi = frame.clone();
            //roi = frame.submat(roiRect); // Use this is want to specify exact area on a frame for detection (helps to speedup processing)

            frameCount++;
            long startTime = System.currentTimeMillis();

            // Run detection
            List<Detection> detections = infer(net, roi, COCO_CLASS_NAMES.size());

            // Draw results
            drawDetections(roi, detections);

            long processingTime = System.currentTimeMillis() - startTime;
            String fpsText = String.format("FPS: %.2f | Frame: %d | Detections: %d",
                    1000.0 / processingTime, frameCount, detections.size());

            // Display bounding boxes
            HighGui.imshow("ROI", roi);
            HighGui.waitKey(1); // Set to 30000 if need to wait for key press

            System.out.println(fpsText);
        }

        // Cleanup
        cap.release();
        frame.release();
        roi.release();
        HighGui.destroyAllWindows();

        System.out.println("** Processing complete **");
    }

}