g++ Main.cpp -L/usr/local/lib -L/usr/local/cuda-6.0/lib `pkg-config --cflags --libs opencv` -o balloonrec

#nvcc -ccbin g++ Main.cpp -I/usr/local/include/opencv -I/usr/local/include  -L/usr/local/lib/ -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres /usr/local/lib/libopencv_ts.a -lopencv_video -lopencv_videostab -lrt -lpthread -lm -ldl -lstdc++ -o balloonrec
