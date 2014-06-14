#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "stdio.h"
#include <sys/time.h>
#include <cmath>

using namespace cv;

#define FEED_SIZE 3
#define PER_FRAME_TIME_LOGGING 0
#define SHOW_FEED_WINDOW 0
#define SHOW_OUTPUT_WINDOW 0
#define SHOW_OTHER_WINDOWS 0
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

void recordTime(long delta, double *avgTime) {
  *avgTime = (*avgTime * nFrames + delta) / (nFrames + 1);
}

long getTimeDelta(struct timeval timea, struct timeval timeb) {
  return 1000000 * (timeb.tv_sec - timea.tv_sec) +
    (int(timeb.tv_usec) - int(timea.tv_usec));
}

int main() {
  struct timeval timea, timeb, startTime, endTime;
  gettimeofday(&startTime, NULL);
  printf("initializing\n");
  Mat frame_host, gray, hsv_host, hue_host, sat_host, val_host, huered_host, scalehuered_host, scalesat_host,
    balloonyness_host, thresh_host, draw;
  gpu::GpuMat frame, hsv, hue, sat, val, huered, scalehuered, scalesat, balloonyness, thresh;
  vector<gpu::GpuMat> hsvplanes(3);
  hsvplanes[0] = hue;
  hsvplanes[1] = sat;
  hsvplanes[2] = val;
  vector< vector< Point > > contours;
  VideoCapture camera(0);
  camera.set(CV_CAP_PROP_FRAME_WIDTH, FEED_WIDTH);
  camera.set(CV_CAP_PROP_FRAME_HEIGHT, FEED_HEIGHT);
  //VideoWriter writer("output.mjpg", CV_FOURCC('M', 'J', 'P', 'G'), 3.5, Size(FEED_WIDTH, FEED_HEIGHT));
  // printf("writer opened: %d\n", writer.isOpened());
  printf("optimized code: %d\n", useOptimized());
  printf("cuda devices: %d\n", gpu::getCudaEnabledDeviceCount());
  printf("current device: %d\n", gpu::getDevice());
  //gpu::setDevice(0);
  int key;
#if (SHOW_FEED_WINDOW == 1)
  namedWindow("feed");
#endif
  //namedWindow("gray");
#if (SHOW_OTHER_WINDOWS == 1)
  namedWindow("hue");
  namedWindow("huered");
  namedWindow("sat");
  namedWindow("val");
  namedWindow("balloonyness");
  namedWindow("threshold");
#endif
#if (SHOW_OUTPUT_WINDOW == 1)
  namedWindow("draw");
#endif
  printf("starting balloon recognition\n");
  //while(true) {
  for (int n = 0; n<50; ++n){
    gettimeofday(&timea, NULL);
    camera >> frame_host;
    draw = frame_host.clone();
    frame.upload(frame_host);
    gettimeofday(&timeb, NULL);
    captureTime = getTimeDelta(timea, timeb);
#if (PER_FRAME_TIME_LOGGING == 1)
    printf("capture frame time used:\t%ld\n", captureTime);
#endif
    //cvtColor(frame, gray, CV_BGR2GRAY);
    // imshow("gray", gray);
    gettimeofday(&timea, NULL);
    gpu::cvtColor(frame, hsv, CV_BGR2HSV);
    //hsv.download(hsv_host);
    gettimeofday(&timeb, NULL);
    conversionTime = getTimeDelta(timea, timeb);
#if (PER_FRAME_TIME_LOGGING == 1)
    printf("color conversion time used:\t%ld\n", conversionTime);
#endif
    gettimeofday(&timea, NULL);
    gpu::split(hsv, hsvplanes);
    hue = hsvplanes[0];
    sat = hsvplanes[1];
    val = hsvplanes[2];
    gettimeofday(&timeb, NULL);
    splitTime = getTimeDelta(timea, timeb);
#if (PER_FRAME_TIME_LOGGING == 1)
    printf("split planes time used:   \t%ld\n", splitTime);
#endif
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
    drawContours(draw, contours, -1, Scalar(255, 0, 0));
#endif
    vector< Point2f > circleCenters(contours.size());
    vector< float > circleRadii(contours.size());
    Point2f center;
    float radius;
    for (int n = 0; n < contours.size(); ++n) {
      minEnclosingCircle(contours[n], center, radius);
#if (DRAW_DEBUG_DATA == 1)
      circle(draw, center, radius, Scalar(0, 255, 255));
#endif
      if (contourArea(contours[n]) >= areaRatio * radius*radius*3.1415926) {
	circle(draw, center, radius, Scalar(0, 255, 0), 2);
      }
    }
    gettimeofday(&timeb, NULL);
    processingTime = getTimeDelta(timea, timeb);
#if (PER_FRAME_TIME_LOGGING == 1)
    printf("frame processing time used:\t%ld\n", processingTime);
#endif
    gettimeofday(&timea, NULL);
#if (SHOW_FEED_WINDOW == 1)
    imshow("feed", frame);
#endif
#if (SHOW_OTHER_WINDOWS == 1)
    imshow("huered", huered);
    imshow("hue", hue);
    imshow("sat", hsvplanes[1]);
    imshow("val", hsvplanes[2]);
    imshow("balloonyness", balloonyness);
    imshow("threshold", thresh);
#endif
#if (SHOW_OUTPUT_WINDOW == 1)
    imshow("draw", draw);
#endif
    //writer << draw;
    gettimeofday(&timeb, NULL);
    displayTime = getTimeDelta(timea, timeb);
#if (PER_FRAME_TIME_LOGGING == 1)
    printf("display frame time used:\t%ld\n", displayTime);
#endif
    recordTime(captureTime, &avgCaptureTime);
    recordTime(conversionTime, &avgConversionTime);
    recordTime(splitTime, &avgSplitTime);
    recordTime(processingTime, &avgProcessingTime);
    recordTime(displayTime, &avgDisplayTime);
    ++nFrames;
    key = waitKey(30);
    if (key >= 0) {
      break;
    }
  }
  gettimeofday(&endTime, NULL);
  printf("key press %d detected. printing statistics.\n", key);
  long totalTimeUsec = getTimeDelta(startTime, endTime);
  double totalTimeSec = double(totalTimeUsec)/1000000.0;
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
