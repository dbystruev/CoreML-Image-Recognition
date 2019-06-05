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

    @IBOutlet weak var descriptionLabel: UILabel!
    
    var captureSession = AVCaptureSession()
    var videoPreviewLayer: AVCaptureVideoPreviewLayer?
    
    var mlModel = Inceptionv3()

    override func viewDidLoad() {
        super.viewDidLoad()
        configure()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        captureSession.startRunning()
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        captureSession.stopRunning()
    }
    
    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
        videoPreviewLayer?.frame.size = size
    }
    
    func configure() {
        guard let captureDevice = AVCaptureDevice.default(for: .video) else {
            print(#function, "Can't get capture device")
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: captureDevice)
            
            captureSession.addInput(input)
            
            let videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "imageRecognition.queue"))
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            captureSession.addOutput(videoDataOutput)
        } catch {
            print(#function, error.localizedDescription)
            return
        }
        
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        videoPreviewLayer?.videoGravity = .resizeAspectFill
        videoPreviewLayer?.frame = view.layer.bounds
        view.layer.addSublayer(videoPreviewLayer!)
        
        descriptionLabel.text = "Looking for objects..."
        view.bringSubviewToFront(descriptionLabel)
    }
    
//    func prediction(image: CVPixelBuffer) throws -> Inceptionv3Output {
//        let input = Inception3Input(image: image)
//        return try mlModel.prediction(input: input)
//    }
    
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        connection.videoOrientation = .portrait
        
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        
        let image = UIImage(ciImage: ciImage)
        
        UIGraphicsBeginImageContext(CGSize(width: 299, height: 299))
            image.draw(in: CGRect(x: 0, y: 0, width: 299, height: 299))
            let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
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
        
        if let pixelBuffer = pixelBuffer {
            guard let output = try? mlModel.prediction(image: pixelBuffer) else {
                return
            }
            
            DispatchQueue.main.async {
                self.descriptionLabel.text = output.classLabel
            }
            
            for (key, value) in output.classLabelProbs {
                print("\(key) = \(value)")
            }
        }
    }
}
