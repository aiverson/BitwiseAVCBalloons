#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <sys/time.h>
#include <cstdio>
#include <cmath>
#include <cstdarg>

using namespace cv;

#define FEED_SIZE 4
#define PER_FRAME_TIME_LOGGING 0
#define SHOW_FEED_WINDOW 0
#define SHOW_OTHER_WINDOWS 0
#define SHOW_OUTPUT_WINDOW 1
#define DRAW_DEBUG_DATA 0

#if (FEED_SIZE == 1)

const int FEED_WIDTH = 320;
const int FEED_HEIGHT = 240;

#endif

#if (FEED_SIZE == 2)

const int FEED_WIDTH = 640;
const int FEED_HEIGHT = 480;

#endif

#if (FEED_SIZE == 3)

const int FEED_WIDTH = 1280;
const int FEED_HEIGHT = 960;

#endif

#if (FEED_SIZE == 4)

const int FEED_WIDTH = 1920;
const int FEED_HEIGHT = 1080;

#endif

double avgCaptureTime = 0,
  avgConversionTime = 0,
  avgSplitTime = 0,
  avgProcessingTime = 0,
  avgDisplayTime = 0;

long captureTime = 0,
  conversionTime = 0,
  splitTime = 0,
  processingTime = 0,
  displayTime = 0;

int nFrames = 0;

const double areaRatio = 0.65;

void initGUI() {
#if (SHOW_FEED_WINDOW == 1)
  namedWindow("feed");
#endif
#if (SHOW_OTHER_WINDOWS == 1)
  namedWindow("hue");
  namedWindow("sat");
  namedWindow("val");
  namedWindow("balloonyness");
#endif
#if (SHOW_OUTPUT_WINDOW == 1)
  namedWindow("debugOverlay");
#endif
}

void recordTime(long delta, double *avgTime) {
  *avgTime = (*avgTime * nFrames + delta) / (nFrames + 1);
}

long getTimeDelta(struct timeval timea, struct timeval timeb) {
  return 1000000 * (timeb.tv_sec - timea.tv_sec) +
    (int(timeb.tv_usec) - int(timea.tv_usec));
}

void log(const char* msg, ...) {
#if (PER_FRAME_TIME_LOGGING == 1)
  va_list args;
  va_start(args, msg);
  printf(msg, args);
#endif
}

void captureFrame(VideoCapture &camera, Mat &frame_host, gpu::GpuMat &frame, Mat &debugOverlay) {
  struct timeval timea, timeb;

  gettimeofday(&timea, NULL);
  camera >> frame_host;
  debugOverlay = frame_host.clone();
  frame.upload(frame_host);
  gettimeofday(&timeb, NULL);

  captureTime = getTimeDelta(timea, timeb);
  log("capture frame time used:\t%ld\n", captureTime);
}

void convertToHSV(gpu::GpuMat &frame, gpu::GpuMat &hue, gpu::GpuMat &sat, gpu::GpuMat &val) {
  struct timeval timea, timeb;
  gpu::GpuMat hsv;

  vector<gpu::GpuMat> hsvplanes(3);
  hsvplanes[0] = hue;
  hsvplanes[1] = sat;
  hsvplanes[2] = val;

  gettimeofday(&timea, NULL);
  gpu::cvtColor(frame, hsv, CV_BGR2HSV);
  gettimeofday(&timeb, NULL);

  conversionTime = getTimeDelta(timea, timeb);
  log("color conversion time used:\t%ld\n", conversionTime);

  gettimeofday(&timea, NULL);
  gpu::split(hsv, hsvplanes);
  hue = hsvplanes[0];
  sat = hsvplanes[1];
  val = hsvplanes[2];
  gettimeofday(&timeb, NULL);

  splitTime = getTimeDelta(timea, timeb);
  log("split planes time used:   \t%ld\n", splitTime);
}

void processFrame(gpu::GpuMat &hue, gpu::GpuMat &sat, gpu::GpuMat &balloonyness, Mat &debugOverlay) {
  struct timeval timea, timeb;
  gpu::GpuMat huered, scalehuered, scalesat, thresh;
  Mat thresh_host;
  vector< vector< Point > > contours;

  gettimeofday(&timea, NULL);

  gpu::absdiff(hue, Scalar(90), huered);
  gpu::divide(huered, Scalar(16), scalehuered);
  gpu::divide(sat, Scalar(64), scalesat);
  gpu::multiply(huered, Scalar(2), scalehuered);
  gpu::multiply(scalehuered, scalesat, balloonyness);
  gpu::threshold(balloonyness, thresh, 200, 255, THRESH_BINARY);
  thresh.download(thresh_host);

  findContours(thresh_host, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

#if (DRAW_DEBUG_DATA == 1)
  drawContours(debugOverlay, contours, -1, Scalar(255, 0, 0));
#endif

  vector< Point2f > circleCenters(contours.size());
  vector< float > circleRadii(contours.size());
  Point2f center;
  float radius;
  for (int n = 0; n < contours.size(); ++n) {
    minEnclosingCircle(contours[n], center, radius);

#if (DRAW_DEBUG_DATA == 1)
    circle(debugOverlay, center, radius, Scalar(0, 255, 255));
#endif

    if (contourArea(contours[n]) >= areaRatio * radius*radius*3.1415926) {
      circle(debugOverlay, center, radius, Scalar(0, 255, 0), 2);
    }
  }

  gettimeofday(&timeb, NULL);
  processingTime = getTimeDelta(timea, timeb);
  log("frame processing time used:\t%ld\n", processingTime);
}

void displayOutput(Mat frame, gpu::GpuMat hue, gpu::GpuMat sat, gpu::GpuMat val, gpu::GpuMat balloonyness, Mat debugOverlay) {
  struct timeval timea, timeb;

  gettimeofday(&timea, NULL);

#if (SHOW_FEED_WINDOW == 1)
  imshow("feed", frame);
#endif
#if (SHOW_OTHER_WINDOWS ==1)
  Mat hue_host, sat_host, val_host, balloonyness_host;
  hue.download(hue_host);
  sat.download(sat_host);
  val.download(val_host);
  balloonyness.download(balloonyness_host);
  imshow("hue", hue_host);
  imshow("sat", sat_host);
  imshow("val", val_host);
  imshow("balloonyness", balloonyness_host);
#endif
#if (SHOW_OUTPUT_WINDOW == 1)
  imshow("debugOverlay", debugOverlay);
#endif

  gettimeofday(&timeb, NULL);
  displayTime = getTimeDelta(timea, timeb);
  log("display frame time used:\t%ld\n", displayTime);
}

int main() {
  struct timeval timea, timeb, startTime, endTime;
  gettimeofday(&startTime, NULL);

  Mat frame_host, thresh_host, debugOverlay;
  gpu::GpuMat frame, hsv, hue, sat, val, huered, scalehuered, scalesat, balloonyness, thresh;
  

  VideoCapture camera(0);
  camera.set(CV_CAP_PROP_FRAME_WIDTH, FEED_WIDTH);
  camera.set(CV_CAP_PROP_FRAME_HEIGHT, FEED_HEIGHT);

  log("optimized code: %d\n", useOptimized());
  log("cuda devices: %d\n", gpu::getCudaEnabledDeviceCount());
  log("current device: %d\n", gpu::getDevice());

  initGUI();
  log("starting balloon recognition\n");

  while(true) {
    captureFrame(camera, frame_host, frame, debugOverlay);
    convertToHSV(frame, hue, sat, val);
    processFrame(hue, sat, balloonyness, debugOverlay);
    displayOutput(frame_host, hue, sat, val, balloonyness, debugOverlay);

    recordTime(captureTime, &avgCaptureTime);
    recordTime(conversionTime, &avgConversionTime);
    recordTime(splitTime, &avgSplitTime);
    recordTime(processingTime, &avgProcessingTime);
    recordTime(displayTime, &avgDisplayTime);

    ++nFrames;

    if (waitKey(30) >= 0) {
      break;
    }
  }

  gettimeofday(&endTime, NULL);
  long totalTimeUsec = getTimeDelta(startTime, endTime);
  double totalTimeSec = double(totalTimeUsec)/1000000.0;

  printf("key press detected. printing statistics.\n");
  printf("%d frames captured over %ld microseconds (%lf seconds)\n", nFrames,
	 totalTimeUsec, totalTimeSec);
  printf("ran at %lf Frames per Second\n", nFrames/totalTimeSec);
  printf("average capture frame time used:\t%lf\n", avgCaptureTime);
  printf("average color conversion time used:\t%lf\n", avgConversionTime);
  printf("average split planes time used:  \t%lf\n", avgSplitTime);
  printf("average frame processing time used:\t%lf\n", avgProcessingTime);
  printf("average display frame time used:\t%lf\n", avgDisplayTime);
  printf("terminating...\n");
}
