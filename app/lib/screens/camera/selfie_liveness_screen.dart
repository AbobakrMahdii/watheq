import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../../ui/widgets/app_snackbars.dart';

class SelfieCaptureResult {
  const SelfieCaptureResult({required this.file, required this.livenessData});

  final File file;
  final Map<String, dynamic> livenessData;
}

class SelfieLivenessScreen extends StatefulWidget {
  const SelfieLivenessScreen({super.key});

  @override
  State<SelfieLivenessScreen> createState() => _SelfieLivenessScreenState();
}

class _SelfieLivenessScreenState extends State<SelfieLivenessScreen> {
  CameraController? _controller;
  bool _isLoading = true;
  bool _isProcessing = false;
  bool _leftDone = false;
  bool _rightDone = false;
  bool _blinkDone = false;
  DateTime? _startTime;
  FaceDetector? _faceDetector;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      final frontCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      _controller = CameraController(
        frontCamera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _controller!.initialize();
      _faceDetector = FaceDetector(
        options: FaceDetectorOptions(
          enableClassification: true,
          enableLandmarks: true,
          enableTracking: true,
        ),
      );
      _startTime = DateTime.now();
      await _controller!.startImageStream(_processCameraImage);
    } catch (e) {
      debugPrint('Camera error: $e');
    }

    if (mounted) {
      setState(() => _isLoading = false);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _faceDetector?.close();
    super.dispose();
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isProcessing || _faceDetector == null) return;
    _isProcessing = true;

    try {
      final inputImage = _inputImageFromCamera(image, _controller!.description.sensorOrientation);
      final faces = await _faceDetector!.processImage(inputImage);
      if (faces.isNotEmpty) {
        final face = faces.first;
        final angleY = face.headEulerAngleY ?? 0;
        final leftEye = face.leftEyeOpenProbability ?? 1;
        final rightEye = face.rightEyeOpenProbability ?? 1;

        if (!_leftDone && angleY < -15) _leftDone = true;
        if (!_rightDone && angleY > 15) _rightDone = true;
        if (!_blinkDone && leftEye < 0.3 && rightEye < 0.3) _blinkDone = true;

        if (_leftDone && _rightDone && _blinkDone) {
          await _finishCapture();
        }
      }
    } catch (_) {}
    _isProcessing = false;
    if (mounted) setState(() {});
  }

  Future<void> _finishCapture() async {
    if (_controller == null) return;
    await _controller!.stopImageStream();
    final photo = await _controller!.takePicture();
    final duration = DateTime.now().difference(_startTime ?? DateTime.now()).inMilliseconds;
    final result = SelfieCaptureResult(
      file: File(photo.path),
      livenessData: {
        "passed": true,
        "left": _leftDone,
        "right": _rightDone,
        "blink": _blinkDone,
        "duration_ms": duration,
      },
    );
    if (!mounted) return;
    Navigator.pop(context, result);
  }

  InputImage _inputImageFromCamera(CameraImage image, int rotation) {
    final bytes = _concatenatePlanes(image.planes);
    final metadata = InputImageMetadata(
      size: Size(image.width.toDouble(), image.height.toDouble()),
      rotation: _rotationFromSensor(rotation),
      format: InputImageFormat.yuv420,
      bytesPerRow: image.planes.first.bytesPerRow,
    );
    return InputImage.fromBytes(bytes: bytes, metadata: metadata);
  }

  InputImageRotation _rotationFromSensor(int rotation) {
    switch (rotation) {
      case 90:
        return InputImageRotation.rotation90deg;
      case 180:
        return InputImageRotation.rotation180deg;
      case 270:
        return InputImageRotation.rotation270deg;
      case 0:
      default:
        return InputImageRotation.rotation0deg;
    }
  }

  Uint8List _concatenatePlanes(List<Plane> planes) {
    final allBytes = BytesBuilder(copy: false);
    for (final plane in planes) {
      allBytes.add(plane.bytes);
    }
    return allBytes.toBytes();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading || _controller == null) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text('تصوير السيلفي')),
      body: Stack(
        children: [
          CameraPreview(_controller!),
          Positioned(
            left: 16,
            right: 16,
            bottom: 24,
            child: Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _StatusRow(label: 'حرّك الرأس يسار', done: _leftDone),
                    _StatusRow(label: 'حرّك الرأس يمين', done: _rightDone),
                    _StatusRow(label: 'ارمش العين', done: _blinkDone),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          AppSnackbars.info(context, 'اتبع التعليمات لإكمال التحقق');
        },
        child: const Icon(Icons.info_outline),
      ),
    );
  }
}

class _StatusRow extends StatelessWidget {
  const _StatusRow({required this.label, required this.done});

  final String label;
  final bool done;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(
          done ? Icons.check_circle : Icons.radio_button_unchecked,
          color: done ? Colors.green : Colors.grey,
          size: 18,
        ),
        const SizedBox(width: 8),
        Expanded(child: Text(label)),
      ],
    );
  }
}
