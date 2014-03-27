//
//  main.cpp
//  Test
//
//  Created by Aleksander Grzyb on 18/03/14.
//  Copyright (c) 2014 PIRO. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);
void loadAssignmentOne(void);
void loadFaces(vector<vector<Mat>> *faces);
void printImageSize(Mat &image);

const string face_cascade_name = "/Users/AleksanderGrzyb/Documents/Studia/Semestr 8/Przetwarzanie i Rozpoznawanie Obrazow/Programy/Zadanie 1/FaceDetection/FaceDetection/haarcascade_frontalface_alt.xml";
const string eyes_cascade_name = "/Users/AleksanderGrzyb/Documents/Studia/Semestr 8/Przetwarzanie i Rozpoznawanie Obrazow/Programy/Zadanie 1/FaceDetection/FaceDetection/haarcascade_eye_tree_eyeglasses.xml";
string pathToFiles = "/Users/AleksanderGrzyb/Documents/Studia/Semestr 8/Przetwarzanie i Rozpoznawanie Obrazow/Programy/Zadanie 1/FaceDetection/faces/";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

int main(int argc, const char** argv)
{
    vector<vector<Mat>> faces;
    vector<Mat> facesToLearn;
    vector<Mat> facesToTest;
    vector<int> labels;
    Ptr<FaceRecognizer> faceRecognizer = createEigenFaceRecognizer();
    
    // Loading faces images from files
    loadFaces(&faces);
    
    double result = 0;
    for (int i = 0; i < 10; i++) {
        int chosenFaceToTest = i;
        facesToTest.clear();
        facesToLearn.clear();
        labels.clear();
        for (int a = 0; a < faces.size(); a++) {
            for (int b = 0; b < faces[i].size(); b++) {
                if (b != chosenFaceToTest) {
                    facesToLearn.push_back(faces[a][b]);
                    labels.push_back(a);
                }
                else {
                    facesToTest.push_back(faces[a][b]);
                }
            }
        }
        faceRecognizer->train(facesToLearn, labels);
        double numberOfCorrectRecognizedFaces = 0;
        for (int a = 0; a < facesToTest.size(); a ++) {
            int prediction = -1;
            double confidence = 0;
            faceRecognizer->predict(facesToTest[a], prediction, confidence);
            if (prediction == a) {
                numberOfCorrectRecognizedFaces++;
            }
        }
        result += numberOfCorrectRecognizedFaces / (double)facesToTest.size();
        cout << "Case: " << numberOfCorrectRecognizedFaces / (double)facesToTest.size() << endl;
    }
    cout << "Final result: " << result / 10.0 << endl;
    return 0;
}

void printImageSize(Mat &image)
{
    cout << "W: " << image.size().width << " H: " << image.size().height << endl;
}

void loadFaces(vector<vector<Mat>> *faces)
{
    for (int i = 0; i < 22; i++) {
        vector<Mat> facesPerPerson;
        string personNumber = "";
        if (i < 9) {
            personNumber += "0";
            personNumber += to_string(i + 1);
        }
        else {
            personNumber += to_string(i + 1);
        }
        for (int a = 0; a < 10; a ++) {
            string faceNumber = to_string(a + 1);
            string currentImage = personNumber;
            if (a < 9) {
                currentImage += "_0";
            }
            else {
                currentImage += "_";
            }
            currentImage += faceNumber;
            currentImage += ".png";
            string imagePath = "";
            imagePath += pathToFiles;
            imagePath += currentImage;
            Mat currentMatImage = imread(imagePath);
            cvtColor(currentMatImage, currentMatImage, CV_BGR2GRAY);
            resize(currentMatImage, currentMatImage, Size(50, 50));
            facesPerPerson.push_back(currentMatImage);
        }
        faces->push_back(facesPerPerson);
        facesPerPerson.clear();
    }
}

void loadAssignmentOne(void)
{
    CvCapture* capture;
    Mat frame;
    FileStorage fs(face_cascade_name, FileStorage::READ);
    if (!fs.isOpened()) {
        cout<<"can not read xml"<<endl;
    }
    if(!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading\n");
//        return -1;
    };
    if(!eyes_cascade.load( eyes_cascade_name)) {
        printf("--(!)Error loading\n");
//        return -1;
    };
    capture = cvCaptureFromCAM(-1);
    if(capture) {
        while(true) {
            frame = cvQueryFrame(capture);
            if(!frame.empty()) {
                detectAndDisplay(frame);
            }
            else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }
            int c = waitKey(10);
            if((char)c == 'c') {
                break;
            }
        }
    }
}

void detectAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
    for(size_t i = 0; i < faces.size(); i++) {
        Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
        ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
        Mat faceROI = frame_gray(faces[i]);
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        for(size_t j = 0; j < eyes.size(); j++) {
            Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5, faces[i].y + eyes[j].y + eyes[j].height * 0.5);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0);
        }
    }
    imshow(window_name, frame);
}
