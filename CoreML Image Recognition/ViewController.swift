//
//  ViewController.swift
//  CoreML Image Recognition
//
//  Created by Denis Bystruev on 25/02/2019.
//  Copyright © 2019 Denis Bystruev. All rights reserved.
//

import AVFoundation
import UIKit

class ViewController: UIViewController {
    
    // MARK: - ... @IBOutlet
    /// Label which will hold the most likely object recognized
    @IBOutlet weak var descriptionLabel: UILabel!
    
    /// Capture activity manager and coordinator
    var captureSession = AVCaptureSession()
    
    /// Core animation layer to display captured video
    var videoPreviewLayer: AVCaptureVideoPreviewLayer?
    
    /// Prediction model from https://developer.apple.com/machine-learning/build-run-models/
    var mlModel = Inceptionv3()
    
    // MARK: - ... UIViewController Methods
    // Configure the capturing when loaded
    override func viewDidLoad() {
        super.viewDidLoad()
        configure()
    }
    
    // Start capturing session when view appears
    override func viewDidAppear(_ animated: Bool) {
        captureSession.startRunning()
    }
    
    // Stop capturing session when view disappears
    override func viewDidDisappear(_ animated: Bool) {
        captureSession.stopRunning()
    }
    
    // MARK: - ... Custom Methods
    /// Configure the capturing
    func configure() {
        // Get the default device for capturing video
        guard let captureDevice = AVCaptureDevice.default(for: .video) else {
            print("ERROR in \(#function) at line \(#line): Can't get capture device")
            return
        }
        
        // Get input from video to capture session
        do {
            let input = try AVCaptureDeviceInput(device: captureDevice)
            
            captureSession.addInput(input)
            
            let videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "imageRecognition.queue"))
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            captureSession.addOutput(videoDataOutput)
        } catch {
            print("ERROR in \(#function) at line \(#line): \(error.localizedDescription)")
            return
        }
        
        // Setup video preview layer
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        videoPreviewLayer?.videoGravity = .resizeAspectFill
        videoPreviewLayer?.frame = view.layer.bounds
        view.layer.addSublayer(videoPreviewLayer!)
        
        // Setup description label
        descriptionLabel.text = "Looking for objects..."
        view.bringSubviewToFront(descriptionLabel)
    }
}

// MARK: - ... AVCaptureVideoDataOutputSampleBufferDelegate
extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        connection.videoOrientation = .portrait
        
        // Resize the frame to 299x299 as required by inceptionv3 model
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        
        let image = UIImage(ciImage: ciImage)
        
        UIGraphicsBeginImageContext(CGSize(width: 299, height: 299))
            image.draw(in: CGRect(x: 0, y: 0, width: 299, height: 299))
            let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        // Convert UIImage to CVPixelBuffer
        // See https://stackoverflow.com/questions/44462087/how-to-convert-a-uiimage-to-a-cvpixelbuffer
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(resizedImage.size.width),
            Int(resizedImage.size.height),
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess else { return }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: pixelData,
            width: Int(resizedImage.size.width),
            height: Int(resizedImage.size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        context?.translateBy(x: 0, y: resizedImage.size.height)
        context?.scaleBy(x: 1, y: -1)
        
        UIGraphicsPushContext(context!)
        resizedImage.draw(in: CGRect(x: 0, y: 0, width: resizedImage.size.width, height: resizedImage.size.height))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        // Pass pixel buffer to Core ML model for predictions
        if let pixelBuffer = pixelBuffer {
            guard let output = try? mlModel.prediction(image: pixelBuffer) else {
                return
            }
            
            // Update the label with most likely category
            DispatchQueue.main.async {
                self.descriptionLabel.text = output.classLabel
            }
            
            // Print all other categories detected
            for (key, value) in output.classLabelProbs {
                print("\(key) = \(value)")
            }
        }
    }
}
